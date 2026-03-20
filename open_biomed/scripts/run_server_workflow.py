import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import asyncio

# import function
from open_biomed.core.llm_provider import ReportGeneratorSBDD, ReportGeneratorGeneral

app = FastAPI()

# Define the request body model
class ReportRequest(BaseModel):
    task: str
    workflow: str
    user_email: str
    num_repeats: int

@app.post("/run_workflow/")
async def run_workflow(request: ReportRequest):
    request = request.model_dump()
    task = request["task"].lower()
    try:
        if task == "drug_design":
            requester = ReportGeneratorSBDD()
        else:
            requester = ReportGeneratorGeneral()
        
        required_inputs = ["workflow", "user_email", "num_repeats"]
        if not all(key in request for key in required_inputs):
            raise HTTPException(
                status_code=400, detail="workflow config file is required for report generation task")
    
        asyncio.create_task(requester.run(workflow=request["workflow"], user_email= request["user_email"], num_repeats=request["num_repeats"]))

        return {"type": task+"_report", "content": "Workflow is still running...", "thinking": ""}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/healthz")
def ping():
    return "Service available"
    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8083, limit_concurrency=2)
