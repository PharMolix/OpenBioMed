"""
BoltzGen All-Atom Protein Design Tool.

Provides async workflow for BoltzGen design tasks:
1. boltzgen_submit - Submit job, return job_id instantly
2. boltzgen_monitor - Background polling (2 min interval) updates SQLite
3. boltzgen_status - Query local SQLite status
4. boltzgen_download - Download results when succeeded
"""

from typing import Tuple, List, Dict, Any, Optional
import os
import time
import logging
import requests
import json
import sqlite3
import asyncio
import uuid

from open_biomed.tools.base_tool import Tool, serial_exec

logger = logging.getLogger('OpenBioMed')

# API endpoint - configurable via environment variable
BOLTZGEN_API_BASE_URL = os.environ.get(
    "BOLTZGEN_API_BASE_URL",
    "http://172.16.20.44:10002"
)

# SQLite database path for job state persistence
BOLTZGEN_JOBS_DB_PATH = os.environ.get(
    "BOLTZGEN_JOBS_DB_PATH",
    "./tmp/boltzgen_jobs.db"
)


class BoltzGenJobStateManager:
    """
    SQLite-based state management for BoltzGen jobs.

    Schema:
        job_id TEXT PRIMARY KEY  -- local job_id (uuid)
        boltzgen_service_job_id TEXT  -- job_id from BoltzGen service
        status TEXT NOT NULL  -- pending, queued, running, succeeded, failed, cancelled
        protocol TEXT
        output_name TEXT
        yaml_file TEXT NOT NULL
        cif_files TEXT  -- JSON array
        num_designs INTEGER
        budget INTEGER
        queue_position INTEGER
        progress INTEGER
        error_message TEXT
        output_files TEXT  -- JSON array
        created_at TIMESTAMP
        updated_at TIMESTAMP
        completed_at TIMESTAMP
    """

    def __init__(self, db_path: str = BOLTZGEN_JOBS_DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database with schema."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS boltzgen_jobs (
                job_id TEXT PRIMARY KEY,
                boltzgen_service_job_id TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                protocol TEXT,
                output_name TEXT,
                yaml_file TEXT NOT NULL,
                cif_files TEXT,
                num_designs INTEGER DEFAULT 10,
                budget INTEGER DEFAULT 2,
                queue_position INTEGER,
                progress INTEGER DEFAULT 0,
                error_message TEXT,
                output_files TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_boltzgen_status
            ON boltzgen_jobs(status)
        """)

        conn.commit()
        conn.close()
        logger.info(f"BoltzGen job database initialized at {self.db_path}")

    def create_job(
        self,
        yaml_file: str,
        protocol: str = "protein-anything",
        num_designs: int = 10,
        budget: int = 2,
        cif_files: List[str] = None,
        output_name: str = None
    ) -> str:
        """
        Create a pending job record in SQLite.

        Returns:
            local job_id (uuid string)
        """
        job_id = uuid.uuid4().hex[:12]

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO boltzgen_jobs (
                job_id, yaml_file, protocol, num_designs, budget,
                cif_files, output_name, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)
        """, (
            job_id, yaml_file, protocol, num_designs, budget,
            json.dumps(cif_files or []), output_name,
            time.strftime("%Y-%m-%d %H:%M:%S"),
            time.strftime("%Y-%m-%d %H:%M:%S")
        ))

        conn.commit()
        conn.close()

        logger.info(f"Created BoltzGen job {job_id} with status pending")
        return job_id

    def update_status(
        self,
        job_id: str,
        status: str,
        boltzgen_service_job_id: str = None,
        queue_position: int = None,
        progress: int = None,
        error_message: str = None,
        output_files: List[str] = None
    ):
        """Update job status from BoltzGen service."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        update_fields = ["status = ?", "updated_at = ?"]
        update_values = [status, time.strftime("%Y-%m-%d %H:%M:%S")]

        if boltzgen_service_job_id is not None:
            update_fields.append("boltzgen_service_job_id = ?")
            update_values.append(boltzgen_service_job_id)

        if queue_position is not None:
            update_fields.append("queue_position = ?")
            update_values.append(queue_position)

        if progress is not None:
            update_fields.append("progress = ?")
            update_values.append(progress)

        if error_message is not None:
            update_fields.append("error_message = ?")
            update_values.append(error_message)

        if output_files is not None:
            update_fields.append("output_files = ?")
            update_values.append(json.dumps(output_files))

        if status in ["succeeded", "failed", "cancelled"]:
            update_fields.append("completed_at = ?")
            update_values.append(time.strftime("%Y-%m-%d %H:%M:%S"))

        cursor.execute(
            f"UPDATE boltzgen_jobs SET {', '.join(update_fields)} WHERE job_id = ?",
            update_values + [job_id]
        )

        conn.commit()
        conn.close()

        logger.info(f"Updated job {job_id} status to {status}")

    def get_job(self, job_id: str) -> Dict[str, Any]:
        """Get job status from SQLite."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT job_id, boltzgen_service_job_id, status, protocol, output_name,
                   yaml_file, cif_files, num_designs, budget, queue_position,
                   progress, error_message, output_files,
                   created_at, updated_at, completed_at
            FROM boltzgen_jobs WHERE job_id = ?
        """, (job_id,))

        row = cursor.fetchone()
        conn.close()

        if row is None:
            return None

        return {
            "job_id": row[0],
            "boltzgen_service_job_id": row[1],
            "status": row[2],
            "protocol": row[3],
            "output_name": row[4],
            "yaml_file": row[5],
            "cif_files": json.loads(row[6] or "[]"),
            "num_designs": row[7],
            "budget": row[8],
            "queue_position": row[9],
            "progress": row[10] or 0,
            "error_message": row[11],
            "output_files": json.loads(row[12] or "[]"),
            "created_at": row[13],
            "updated_at": row[14],
            "completed_at": row[15]
        }

    def list_active_jobs(self) -> List[str]:
        """Get all jobs with status in [queued, running]."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT job_id FROM boltzgen_jobs
            WHERE status IN ('queued', 'running')
        """)

        rows = cursor.fetchall()
        conn.close()

        return [row[0] for row in rows]


class BoltzGenSubmitTool(Tool):
    """
    Submit BoltzGen design job.

    Instant response with job_id.
    """

    def __init__(self):
        self.state_manager = BoltzGenJobStateManager()
        self.api_base = BOLTZGEN_API_BASE_URL

    def print_usage(self) -> str:
        return "\n".join([
            'BoltzGen Submit - Submit design job',
            'Inputs:',
            '  - yaml_file: Design YAML file path (required)',
            '  - protocol: Design protocol (default: protein-anything)',
            '  - num_designs: Number of intermediate designs (default: 10)',
            '  - budget: Final diversity-optimized set size (default: 2)',
            '  - cif_files: List of CIF/PDB target file paths (optional)',
            '  - output_name: Output file name prefix (optional)',
            'Outputs:',
            '  - job_id: Local job ID',
            '  - status: pending/queued/running',
            '  - boltzgen_service_url: Direct link to BoltzGen service',
        ])

    def _health_check(self) -> bool:
        """Check BoltzGen API availability."""
        try:
            response = requests.get(
                f"{self.api_base}/health",
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"BoltzGen health check failed: {e}")
            return False

    def run(
        self,
        yaml_file: str,
        protocol: str = "protein-anything",
        num_designs: int = 10,
        budget: int = 2,
        cif_files: List[str] = None,
        output_name: str = None,
        **kwargs
    ) -> Tuple[Dict, List[str]]:
        """
        Submit design job to BoltzGen API.

        Returns:
            Tuple of (job_info dict, messages list)
        """
        logger.info(f"Submitting BoltzGen job with protocol: {protocol}")

        if not yaml_file:
            return {}, ["Error: yaml_file is required"]

        # Create local job record
        local_job_id = self.state_manager.create_job(
            yaml_file=yaml_file,
            protocol=protocol,
            num_designs=num_designs,
            budget=budget,
            cif_files=cif_files,
            output_name=output_name
        )

        # Health check
        if not self._health_check():
            logger.warning("BoltzGen API not healthy, job created but not submitted")
            return {
                "job_id": local_job_id,
                "status": "pending",
                "message": "BoltzGen service unavailable. Job pending submission."
            }, [f"Job {local_job_id} created but BoltzGen service unavailable"]

        # Submit to BoltzGen API
        try:
            files = {}
            yaml_file_name = os.path.basename(yaml_file)
            files['design_yaml'] = (yaml_file_name, open(yaml_file, 'rb'))

            if cif_files:
                for cif_path in cif_files:
                    cif_file_name = os.path.basename(cif_path)
                    files.setdefault('files', [])
                    files['files'].append((cif_file_name, open(cif_path, 'rb')))

            data = {
                'protocol': protocol,
                'num_designs': str(num_designs),
                'budget': str(budget)
            }

            response = requests.post(
                f"{self.api_base}/jobs",
                files=files,
                data=data,
                timeout=60
            )

            # Close file handles
            for key, file_tuple in files.items():
                if isinstance(file_tuple, tuple):
                    file_tuple[1].close()
                elif isinstance(file_tuple, list):
                    for ft in file_tuple:
                        ft[1].close()

            if response.status_code == 503:
                self.state_manager.update_status(
                    local_job_id, "pending",
                    error_message="BoltzGen queue is full (max 5 jobs)"
                )
                return {
                    "job_id": local_job_id,
                    "status": "pending",
                    "error": "Queue full"
                }, ["BoltzGen queue is full. Please wait for existing jobs to finish."]

            if response.status_code != 200:
                self.state_manager.update_status(
                    local_job_id, "failed",
                    error_message=f"Submit failed: {response.status_code}"
                )
                return {
                    "job_id": local_job_id,
                    "status": "failed"
                }, [f"Submit failed: {response.status_code}"]

            result = response.json()
            service_job_id = result.get("job_id")
            status = result.get("status", "queued")
            queue_position = result.get("queue_position")

            # Update SQLite with service job_id
            self.state_manager.update_status(
                local_job_id, status,
                boltzgen_service_job_id=service_job_id,
                queue_position=queue_position
            )

            service_url = f"{self.api_base}/jobs/{service_job_id}"

            return {
                "job_id": local_job_id,
                "boltzgen_service_job_id": service_job_id,
                "status": status,
                "queue_position": queue_position,
                "boltzgen_service_url": service_url
            }, [f"Job {local_job_id} submitted to BoltzGen (service job: {service_job_id})"]

        except Exception as e:
            logger.error(f"Submit failed: {e}")
            self.state_manager.update_status(
                local_job_id, "failed",
                error_message=str(e)
            )
            return {
                "job_id": local_job_id,
                "status": "failed"
            }, [f"Submit error: {str(e)}"]


class BoltzGenMonitorTool(Tool):
    """
    Background monitoring tool for BoltzGen jobs.

    Polls BoltzGen API every 2 minutes and updates SQLite.
    """

    def __init__(self):
        self.state_manager = BoltzGenJobStateManager()
        self.api_base = BOLTZGEN_API_BASE_URL
        self.poll_interval = 120  # 2 minutes

    def print_usage(self) -> str:
        return "\n".join([
            'BoltzGen Monitor - Background polling for job status',
            'Inputs:',
            '  - job_id: Specific job to monitor (optional, monitors all active if null)',
            'Outputs:',
            '  - monitoring: List of job_ids being monitored',
            '  - poll_interval: 120 seconds',
            '  - estimated_duration: 12-45 minutes',
        ])

    async def run_async(
        self,
        job_id: str = None,
        **kwargs
    ) -> Tuple[Dict, List[str]]:
        """
        Start background monitoring.

        Returns immediately with monitoring info.
        Actual monitoring runs in background.
        """
        # Get jobs to monitor
        if job_id:
            jobs = [job_id]
        else:
            jobs = self.state_manager.list_active_jobs()

        if not jobs:
            return {
                "monitoring": [],
                "poll_interval": self.poll_interval,
                "message": "No active jobs to monitor"
            }, ["No jobs with status queued/running found"]

        # Start background monitoring for each job
        # Note: This method returns immediately; actual polling happens in background
        # The caller (run_server.py) should use BackgroundTasks to run monitor_single_job

        return {
            "monitoring": jobs,
            "poll_interval": self.poll_interval,
            "estimated_duration": "12-45 minutes",
            "message": "Background monitoring started. Design typically takes 12-45 minutes."
        }, [f"Monitoring {len(jobs)} jobs: {jobs}"]

    def monitor_single_job(self, job_id: str):
        """
        Background task to poll single job status.

        Called by run_server.py via BackgroundTasks.
        This is a synchronous method that uses time.sleep() for polling.
        """
        job = self.state_manager.get_job(job_id)
        if not job:
            logger.warning(f"Job {job_id} not found in SQLite")
            return

        service_job_id = job.get("boltzgen_service_job_id")
        if not service_job_id:
            logger.warning(f"Job {job_id} has no service_job_id")
            return

        logger.info(f"Starting background monitor for job {job_id} (service: {service_job_id})")

        while True:
            try:
                # Poll BoltzGen API
                response = requests.get(
                    f"{self.api_base}/jobs/{service_job_id}",
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()
                    status = result.get("status")
                    progress = result.get("progress", 0)
                    error_message = result.get("error_message")

                    # Update SQLite
                    self.state_manager.update_status(
                        job_id, status,
                        progress=progress,
                        error_message=error_message
                    )

                    logger.info(f"Job {job_id} status: {status}, progress: {progress}%")

                    # Stop monitoring if completed
                    if status in ["succeeded", "failed", "cancelled"]:
                        logger.info(f"Job {job_id} completed with status: {status}")
                        break

            except Exception as e:
                logger.warning(f"Poll error for job {job_id}: {e}")

            # Wait 2 minutes before next poll (synchronous sleep)
            time.sleep(self.poll_interval)

    def run(self, job_id: str = None, **kwargs) -> Tuple[Dict, List[str]]:
        """Sync wrapper - should use run_async in handler."""
        return {
            "monitoring": [job_id] if job_id else self.state_manager.list_active_jobs(),
            "poll_interval": self.poll_interval,
            "estimated_duration": "12-45 minutes"
        }, ["Use async handler for background monitoring"]


class BoltzGenStatusTool(Tool):
    """
    Query job status from SQLite.

    Fast response (< 100ms), no external API calls.
    """

    def __init__(self):
        self.state_manager = BoltzGenJobStateManager()
        self.api_base = BOLTZGEN_API_BASE_URL

    def print_usage(self) -> str:
        return "\n".join([
            'BoltzGen Status - Query job status from local SQLite',
            'Inputs:',
            '  - job_id: Job ID to query (required)',
            'Outputs:',
            '  - status: pending/queued/running/succeeded/failed/cancelled',
            '  - progress: Estimated progress %',
            '  - error_message: Error if failed',
        ])

    def run(self, job_id: str, **kwargs) -> Tuple[Dict, List[str]]:
        """
        Query job status from SQLite.

        Returns:
            Tuple of (job_info dict, messages list)
        """
        if not job_id:
            return {}, ["Error: job_id is required"]

        job = self.state_manager.get_job(job_id)

        if not job:
            return {}, [f"Error: Job {job_id} not found"]

        service_url = None
        if job.get("boltzgen_service_job_id"):
            service_url = f"{self.api_base}/jobs/{job['boltzgen_service_job_id']}"

        return {
            "job_id": job["job_id"],
            "boltzgen_service_job_id": job.get("boltzgen_service_job_id"),
            "status": job["status"],
            "progress": job.get("progress", 0),
            "protocol": job.get("protocol"),
            "error_message": job.get("error_message"),
            "queue_position": job.get("queue_position"),
            "created_at": job.get("created_at"),
            "updated_at": job.get("updated_at"),
            "completed_at": job.get("completed_at"),
            "boltzgen_service_url": service_url
        }, [f"Job {job_id} status: {job['status']}"]


class BoltzGenDownloadTool(Tool):
    """
    Download BoltzGen results when job completed.

    Only works when status == 'succeeded'.
    """

    def __init__(self, output_dir: str = "./tmp/boltzgen"):
        self.state_manager = BoltzGenJobStateManager()
        self.api_base = BOLTZGEN_API_BASE_URL
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def print_usage(self) -> str:
        return "\n".join([
            'BoltzGen Download - Download results when job succeeded',
            'Inputs:',
            '  - job_id: Job ID (required)',
            'Outputs:',
            '  - output_files: List of downloaded file paths',
            '  - description: Summary of results',
        ])

    def run(self, job_id: str, **kwargs) -> Tuple[List[str], List[str]]:
        """
        Download results from BoltzGen API.

        Returns:
            Tuple of (file_paths list, messages list)
        """
        if not job_id:
            return [], ["Error: job_id is required"]

        job = self.state_manager.get_job(job_id)

        if not job:
            return [], [f"Error: Job {job_id} not found"]

        if job["status"] != "succeeded":
            return [], [f"Error: Job status is {job['status']}, not succeeded"]

        service_job_id = job.get("boltzgen_service_job_id")
        if not service_job_id:
            return [], ["Error: No service_job_id for this job"]

        output_name = job.get("output_name") or f"boltzgen_{job_id}"
        job_output_dir = os.path.join(self.output_dir, job_id)
        os.makedirs(job_output_dir, exist_ok=True)

        output_files = []

        try:
            # Download all results as zip
            response = requests.get(
                f"{self.api_base}/jobs/{service_job_id}/download",
                timeout=120,
                stream=True
            )

            if response.status_code == 200:
                zip_path = os.path.join(job_output_dir, "results.zip")
                with open(zip_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                # Unzip
                import zipfile
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(job_output_dir)

                os.remove(zip_path)

                # Collect files
                for root, dirs, files in os.walk(job_output_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        output_files.append(file_path)

                # Update SQLite with output files
                self.state_manager.update_status(job_id, "succeeded", output_files=output_files)

                logger.info(f"Downloaded {len(output_files)} files for job {job_id}")

                description = self._format_description(job, output_files)
                return output_files, [description]

            else:
                return [], [f"Download failed: {response.status_code}"]

        except Exception as e:
            logger.error(f"Download error: {e}")
            return [], [f"Download error: {str(e)}"]

    def _format_description(self, job: Dict, output_files: List[str]) -> str:
        """Format output description."""
        protocol = job.get("protocol", "protein-anything")
        protocol_display = protocol.replace("_", "-").capitalize()

        desc_parts = [
            f"BoltzGen {protocol_display} design completed.",
            f"Job ID: {job['job_id']}",
            f"Service Job ID: {job.get('boltzgen_service_job_id')}",
            f"Output directory: ./tmp/boltzgen/{job['job_id']}",
            f"Total output files: {len(output_files)}"
        ]

        # Find key files
        for f in output_files:
            if "design.cif" in f:
                desc_parts.append(f"Design structure: {f}")
            if "aggregate_metrics_analyze.csv" in f:
                desc_parts.append(f"Quality metrics: {f}")

        return "\n".join(desc_parts)


# Legacy class removed - boltzgen_structure_design task deprecated
# Use BoltzGenSubmitTool + BoltzGenMonitorTool + BoltzGenStatusTool + BoltzGenDownloadTool instead


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Test state manager
    print("Testing BoltzGenJobStateManager...")
    state_mgr = BoltzGenJobStateManager()

    # Create test job
    test_job_id = state_mgr.create_job(
        yaml_file="./test.yaml",
        protocol="protein-anything"
    )
    print(f"Created test job: {test_job_id}")

    # Get job
    job = state_mgr.get_job(test_job_id)
    print(f"Job status: {job}")

    # Update status
    state_mgr.update_status(test_job_id, "queued", queue_position=2)
    job = state_mgr.get_job(test_job_id)
    print(f"Updated job: {job}")

    # List active
    active = state_mgr.list_active_jobs()
    print(f"Active jobs: {active}")

    print("\nTools available:")
    print("- BoltzGenSubmitTool")
    print("- BoltzGenMonitorTool")
    print("- BoltzGenStatusTool")
    print("- BoltzGenDownloadTool")