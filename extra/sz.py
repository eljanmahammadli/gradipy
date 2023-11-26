import os
import subprocess


def get_repo_size(repo_path, exclude_keywords=None, exclude_folders=None):
    os.chdir(repo_path)

    # 'git ls-files' to get a list of all tracked files in the repository
    git_ls_files = subprocess.Popen(["git", "ls-files"], stdout=subprocess.PIPE)
    files = git_ls_files.communicate()[0].decode("utf-8").splitlines()

    file_details = []
    for file in files:
        # check excluded keywords and folders
        if exclude_folders and any(file.startswith(folder) for folder in exclude_folders):
            continue
        if exclude_keywords and any(keyword in file for keyword in exclude_keywords):
            continue

        # get the line count for each file
        wc_process = subprocess.Popen(["wc", "-l", file], stdout=subprocess.PIPE)
        line_count = int(wc_process.communicate()[0].decode("utf-8").split()[0])
        file_details.append({"file": file, "lines": line_count})

    # sort the file details based on the number of lines
    sorted_file_details = sorted(file_details, key=lambda x: x["lines"], reverse=True)
    print("\n{:<50} {:<10}".format("File", "Lines"))
    print("=" * 56)

    # Print sorted file details
    for details in sorted_file_details:
        print("{:<50} {:<10}".format(details["file"], details["lines"]))
    print("=" * 56)
    total_lines = sum(details["lines"] for details in sorted_file_details)
    return total_lines


if __name__ == "__main__":
    repo_path = "."
    exclude_keywords = [
        ".ipynb",
        "imagenet.py",
        "LICENSE",
        "setup.py",
        ".gitignore",
        ".gitattributes",
        ".gitkeep",
        "README.md",
        "sz.py",
    ]
    exclude_folders = ["models", "examples", ".github", "test"]
    total_lines = get_repo_size(repo_path, exclude_keywords, exclude_folders)

    if total_lines is not None:
        print("{:<50} {:<10}\n".format("gradipy", total_lines))
