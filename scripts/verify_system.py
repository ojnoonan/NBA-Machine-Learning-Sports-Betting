#!/usr/bin/env python3
"""
System verification script for NBA ML project.
This script checks if your system meets the minimum requirements.
"""

import os
import sys
import psutil
import sqlite3
import logging
import platform
from pathlib import Path
import multiprocessing as mp

def check_python_version():
    """Check Python version."""
    current = sys.version_info
    required = (3, 9)
    
    print(f"\n🐍 Python Version:")
    print(f"Required: {required[0]}.{required[1]}+")
    print(f"Current:  {current.major}.{current.minor}")
    
    return current >= required

def check_cpu():
    """Check CPU cores."""
    cores = mp.cpu_count()
    recommended = 8
    
    print(f"\n💻 CPU Cores:")
    print(f"Recommended: {recommended}+")
    print(f"Available:   {cores}")
    
    return cores >= recommended

def check_memory():
    """Check available RAM."""
    total_ram = psutil.virtual_memory().total / (1024 ** 3)  # Convert to GB
    recommended = 32
    
    print(f"\n🧠 RAM:")
    print(f"Recommended: {recommended}+ GB")
    print(f"Available:   {total_ram:.1f} GB")
    
    return total_ram >= recommended

def check_disk_space():
    """Check available disk space."""
    path = Path(__file__).parent.parent
    total, used, free = psutil.disk_usage(path)
    free_gb = free / (1024 ** 3)  # Convert to GB
    recommended = 100
    
    print(f"\n💾 Disk Space:")
    print(f"Recommended: {recommended}+ GB")
    print(f"Available:   {free_gb:.1f} GB")
    
    return free_gb >= recommended

def check_sqlite():
    """Check SQLite version and functionality."""
    try:
        sqlite_version = sqlite3.sqlite_version
        recommended = "3.35.0"
        
        print(f"\n🗄️  SQLite Version:")
        print(f"Recommended: {recommended}+")
        print(f"Current:     {sqlite_version}")
        
        return sqlite_version >= recommended
    except Exception as e:
        print(f"Error checking SQLite: {e}")
        return False

def main():
    """Run all system checks."""
    print("🔍 NBA ML Project System Verification")
    print("=====================================")
    
    checks = {
        "Python Version": check_python_version(),
        "CPU Cores": check_cpu(),
        "RAM": check_memory(),
        "Disk Space": check_disk_space(),
        "SQLite": check_sqlite()
    }
    
    print("\n📊 Summary:")
    print("=========")
    
    all_passed = True
    for check, passed in checks.items():
        status = "✅ Pass" if passed else "❌ Fail"
        print(f"{check}: {status}")
        all_passed = all_passed and passed
    
    print("\n🎯 Final Result:")
    if all_passed:
        print("✨ All checks passed! Your system meets the recommended specifications.")
    else:
        print("⚠️  Some checks failed. Your system might experience performance issues.")
        print("   Please review the requirements in docs/ENHANCED_STATS_PROCESSING.md")

if __name__ == "__main__":
    main()
