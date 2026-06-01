import os
import oss2
from datetime import datetime
from urllib.parse import unquote
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv(".env")

access_key_id = os.getenv("ACCESS_KEY_ID")
access_key_secret = os.getenv("ACCESS_KEY_SECRET")
bucket_name = os.getenv("BUCKET_NAME", "openbiomed")
endpoint = os.getenv("ENDPOINT", "oss-cn-beijing.aliyuncs.com")


class Oss_Warpper:

    def __init__(self):
        # 创建Bucket对象，所有Object相关的接口都可以通过Bucket对象来进行
        self.bucket = oss2.Bucket(
            oss2.Auth(access_key_id, access_key_secret), endpoint, bucket_name)
    
    @staticmethod
    def generate_file_name(local_file_path):
        """
        生成包含日期和时间的文件名
        :param local_file_path: 文件名
        :return: 拼接后的文件名
        """
        file_name = os.path.basename(local_file_path)
        file_extension = os.path.splitext(local_file_path)[-1]
        file_name_without_extension = os.path.splitext(file_name)[0]
        # 获取当前时间并格式化为 YYYYMMDD_HHMMSS
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 拼接文件名
        file_name = f"{file_name_without_extension}_{timestamp}{file_extension}"
        return file_name
    

    def upload(self, oss_file_path, local_file_path) -> str:
        try:
            self.bucket.put_object_from_file(oss_file_path, local_file_path)
            if self.bucket.object_exists(oss_file_path):
                print("文件已成功上传到OSS！")
            else:
                print("文件未上传成功！")
            key = unquote(oss_file_path)
            url = self.bucket.sign_url('GET', key, 10000, slash_safe=True)
            print('签名url的地址为：', url)
            return url
        except Exception as e:
            print(f"上传文件时出错：{e}")
            return ""

    def download(self, oss_file_path, local_file_path):
        # 从OSS下载文件到本地
        if self.bucket.object_exists(oss_file_path):
            self.bucket.get_object_to_file(oss_file_path, local_file_path)
            print(f"文件已成功下载到本地路径：{local_file_path}")
        else:
            print("指定的文件在OSS中不存在！")

    def download_from_url(self, signed_url, local_file_path):
        """
        通过签名URL下载文件到本地
        :param signed_url: 有效的签名URL
        :param local_file_path: 本地保存路径
        """
        try:
            # 发起GET请求
            response = requests.get(signed_url, stream=True)
            response.raise_for_status()  # 检查请求是否成功

            # 将内容保存到本地文件
            with open(local_file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"文件已成功下载到本地路径：{local_file_path}")
        except requests.exceptions.RequestException as e:
            print(f"下载失败：{e}")



oss_warpper = Oss_Warpper()

if __name__ == '__main__':
    oss_warpper = Oss_Warpper()
    local_file_path = "requirements.txt"
    oss_file_path = oss_warpper.generate_file_name(local_file_path)
    oss_warpper.upload(oss_file_path, local_file_path)
    oss_warpper.download('requirements.txt', 'test.txt')


