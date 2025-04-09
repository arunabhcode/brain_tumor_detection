#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
from logger import print  # Import the custom print


class FilesystemUtils:

    def __init__(self):
        self.data_ext = ".jpg"
        # Initialize current_dir here or ensure get_repo_root is called first
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.repo_root = self.get_repo_root()  # Get repo root during initialization
        self.data_dir = os.path.join(self.repo_root, "data")
        self.model_dir = os.path.join(self.repo_root, "bin")

    def get_repo_root(self):
        """
        Get the root directory of the repository.
        """
        # Start from the directory of the current file
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Traverse up the directory tree until we find the root directory (.git)
        while True:
            git_path = os.path.join(current_dir, ".git")
            # Check if .git exists and is a directory (or a file in case of submodules/worktrees)
            if os.path.exists(git_path):
                return current_dir
            parent_dir = os.path.dirname(current_dir)
            # Stop if we have reached the filesystem root
            if parent_dir == current_dir:
                print("Error: Could not find the repository root (.git folder).")
                return "/"  # Or raise an exception
            current_dir = parent_dir
        return "/"  # Should not be reached if .git is found

    def get_data_dir(self):
        """
        Get the data directory.
        """
        if not self.repo_root:
            print("Error: Repository root not found, cannot determine data directory.")
            return None
        return self.data_dir

    def get_model_dir(self):
        """
        Get the model directory.
        """
        if not self.repo_root:
            print("Error: Repository root not found, cannot determine model directory.")
            return None
        return self.model_dir

    def get_data_files(self):
        """
        Get all data files in the data directory.
        """
        data_dir_path = self.get_data_dir()
        if not data_dir_path:
            return []  # Return empty list if data dir not found

        search_pattern = os.path.join(data_dir_path, "**/*" + self.data_ext)
        print(f"Searching for data files in: {search_pattern}")
        data_files = glob.glob(search_pattern, recursive=True)
        return data_files
