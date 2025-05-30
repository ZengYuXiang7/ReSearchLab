# coding : utf-8
# Author : yuxiang Zeng
import pexpect
import subprocess
import sys
import os
import pickle
import requests
import shutil
import os

# 这里要替换自己的token
access_token = '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'

ignore_content = [
    ".DS_Store",
    "/.idea/",
    "**/__pycache__/",
    "/checkpoints/",
    "git.py",
    "/datasets/",
]

def create_github_repo(repo_name, description="", private=True):
    url = "https://api.github.com/user/repos"
    headers = {
        "Authorization": f"token {access_token}",
        "Accept": "application/vnd.github.v3+json"
    }
    data = {"name": repo_name, "description": description, "private": private}
    response = requests.post(url, headers=headers, json=data, proxies={"http": None, "https": None})
    if response.status_code == 201:
        print(f"Repository '{repo_name}' created successfully.")
    else:
        print(f"Failed to create repository: {response.json().get('message')}")


def delete_github_repo(user_name, repo_name):
    url = f"https://api.github.com/repos/{user_name}/{repo_name}"
    headers = {
        "Authorization": f"token {access_token}",
        "Accept": "application/vnd.github.v3+json"
    }
    response = requests.delete(url, headers=headers, proxies={"http": None, "https": None})

    if response.status_code == 204:
        print(f"Remote repository '{repo_name}' deleted successfully.")
    else:
        print(f"Failed to delete remote repository: {response.json().get('message')}")


def git_first_push(user_name, repo_name):
    url = f"https://github.com/{user_name}/{repo_name}.git"

    def run_command(command):
        result = subprocess.run(command, shell=True)
        return result.returncode == 0  # 返回 True 表示成功，False 表示失败

    if not run_command("git init"):
        return False
    if not run_command("git add -A"):
        return False
    if not run_command("git commit -m 'First Commit'"):
        return False
    if not run_command("git branch -M main"):
        return False
    if not run_command(f"git remote add origin {url}"):
        return False
    if not run_command("git config --global credential.helper store"):
        return False
    if not run_command("git push -u origin main"):
        return False

    return True  # 所有命令成功执行，返回 True


def git_push(message):
    subprocess.run(f'git add -A', shell=True)
    subprocess.run(f'git commit -am "{message}"', shell=True)
    subprocess.run(f"git push", shell=True)


def git_pull():
    subprocess.run(f'git pull', shell=True)


def git_update():
    subprocess.run(f'git commit -am "Commit the work before the pull"', shell=True)
    subprocess.run(f'git pull', shell=True)


def git_reset(cnt):
    subprocess.run(f"git branch -m master main", shell=True)


def hidden_this_py_first():

    with open(".gitignore", "w") as gitignore_file:
        for item in ignore_content:
            gitignore_file.write(f"{item}\n")


def git_clone(repo_url, target_dir=None):
    # 使用默认仓库名作为临时克隆目录
    repo_name = repo_url.split('/')[-1].replace('.git', '')
    clone_dir = target_dir or repo_name

    # 克隆命令
    command = f"git clone {repo_url} {clone_dir}"
    result = subprocess.run(command, shell=True)

    if result.returncode == 0:
        print(f"Repository cloned successfully into '{clone_dir}'")

        # 移动克隆文件夹内容到当前目录
        for item in os.listdir(clone_dir):
            s = os.path.join(clone_dir, item)
            d = os.path.join(os.getcwd(), item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)

        # 删除原始克隆的文件夹
        shutil.rmtree(clone_dir)
        print(f"Moved contents of '{clone_dir}' to current directory and deleted '{clone_dir}'")
    else:
        print("Failed to clone repository.")


if __name__ == "__main__":
    hidden_this_py_first()
    inputs = input('create, push, pull, reset, update, or clone? : ').strip()
    if inputs == 'pull':
        git_pull()
    elif inputs == 'push':
        message = input('message : ').strip()
        git_push(message)
    elif inputs == 'reset':
        cnt = int(input('number of commits : ').strip())
        git_reset(cnt)
    elif inputs == 'update':
        git_update()
    elif inputs == 'create':
        repo_name = input('repository name : ').strip()
        description = input('description : ').strip()
        create_github_repo(repo_name, description=description, private=False)
        user_name = input('username : ').strip()
        if not git_first_push(user_name, repo_name):
            delete_github_repo(user_name, repo_name)
            subprocess.run("rm -rf .git", shell=True)
            print("Local Git repository has been deleted.")
    elif inputs == 'clone':
        repo_url = input('Repository URL : ').strip()
        target_dir = input('Target directory (optional) : ').strip()
        git_clone(repo_url, target_dir if target_dir else None)
