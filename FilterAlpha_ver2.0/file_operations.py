"""
file_operations.py
This module provides utility functions and classes for robust file and directory operations,
specifically tailored for managing "alpha" directories and files in a backtesting or research workflow.
It includes safe removal, backup, comparison, transfer, and update mechanisms for directories and files,
with detailed error handling and logging.
Classes:
    FileOperationError: Custom exception for file operation failures.
Functions:
    hash_file(file_path: Path, block_size: int = 65536) -> str
        Compute the SHA256 hash of a file, reading in blocks for efficiency.
    rmtree_alpha_file(path: Path) -> None
        Safely remove a directory and all its contents, with error handling.
    backup_alpha(dst: Path, alpha: str) -> None
        Create a timestamped backup of a destination directory before overwriting.
    fix_nested_path(dst: Path, alpha: str) -> None
        Detect and fix nested directory structures (e.g., dst/alpha/alpha).
    compare_alpha_cls(src: Path, dst: Path, alpha: str) -> bool
        Compare 'alpha_cls.py' files in source and destination directories for equality.
    move_alpha_directory(src: Path, dst: Path, alpha: str) -> None
        Move an alpha directory to a destination, overwriting if necessary.
    transfer_alpha(alpha: str, src: Path, dst: Path, nf_extension: str, out_decay: str) -> None
        Transfer a single alpha directory to the next stage, with optional comparison and cleanup.
    update_total_alpha(alpha: str, backtest_path: Path, update_alpha_path: Path, update_pnl_path: Path,
                      alpha_hdf_path: str, output_pnl_path: str, output_af_path: str) -> None
        Update and synchronize files for a "total extension" alpha, handling HDF5, PNL, and summary files.
Logging:
    Uses the standard Python logging module for info, warning, and error messages throughout all operations.
"""

import os
import shutil
import filecmp
import hashlib
import time
import logging
from pathlib import Path
from typing import Union, List

# Configure logging
logger = logging.getLogger(__name__)

class FileOperationError(Exception):
    """Custom exception for file operation failures.

    This exception is raised when a file or directory operation fails due to
    reasons such as missing files, permission issues, or unexpected errors.

    Use this exception to handle errors in file operations gracefully and
    provide meaningful error messages to the user or logs.

    Example:
        try:
            # Some file operation
            raise FileOperationError("Failed to delete the file.")
        except FileOperationError as e:
            logger.error(f"An error occurred: {e}")
    """
    pass

def hash_file(file_path: Path, block_size: int = 65536) -> str:
    """Compute SHA256 hash of a file, reading in blocks.

    Args:
        file_path (Path): Path to the file.
        block_size (int): Size of blocks to read.

    Returns:
        str: SHA256 hash of the file.

    Raises:
        FileOperationError: If the file cannot be read.
    """
    sha256 = hashlib.sha256()
    try:
        with open(file_path, 'rb') as f:
            while True:
                data = f.read(block_size)
                if not data:
                    break
                sha256.update(data)
        return sha256.hexdigest()
    except FileNotFoundError:
        raise FileOperationError(f"File not found for hashing: {file_path}")
    except Exception as e:
        raise FileOperationError(f"Error hashing file {file_path}: {str(e)}")

def rmtree_alpha_file(path: Path) -> None:
    """Remove directory and its contents safely.

    Args:
        path (Path): Path to the directory.

    Raises:
        FileOperationError: If the directory cannot be removed.
    """
    path = Path(path)
    try:
        shutil.rmtree(path, ignore_errors=False)
    except FileNotFoundError:
        raise FileOperationError(f"Directory not found: {path}")
    except PermissionError:
        logger.error(f"Permission denied: {path}. Current permissions: {oct(path.stat().st_mode)}. "
                     f"Consider checking ownership or modifying permissions.")
        raise FileOperationError(f"Error removing directory {path} ({type(e).__name__}): {str(e)}")
    except Exception as e:
        raise FileOperationError(f"Error removing directory {path}: {str(e)}")

def backup_alpha(dst: Path, alpha: str) -> None:
    """Backup destination directory before overwriting.

    Args:
        dst (Path): Destination directory.
        alpha (str): Alpha name.

    Raises:
        FileOperationError: If the backup fails.
    """
    if dst.exists():
        backup_path = dst.parent / f"{alpha}_backup_{time.strftime('%Y%m%d_%H%M%S')}"
        try:
            shutil.copytree(dst, backup_path)
            logger.info(f"Backed up {dst} to {backup_path}")
        except Exception as e:
            raise FileOperationError(f"Failed to backup {dst} to {backup_path}: {str(e)}")

def fix_nested_path(dst: Path, alpha: str) -> None:
    """Check and fix nested directory structure (e.g., dst/alpha/alpha).

    Args:
        dst (Path): Destination directory.
        alpha (str): Alpha name.

    Raises:
        FileOperationError: If the nested path cannot be fixed.
    """
    nested_path = dst / alpha
    if nested_path.exists() and nested_path.is_dir():
        try:
            for item in nested_path.iterdir():
                shutil.move(str(item), str(dst / item.name))
            rmtree_alpha_file(nested_path)
            logger.info(f"Fixed nested path by elevating contents from {nested_path} to {dst}")
        except Exception as e:
            raise FileOperationError(f"Failed to fix nested path {nested_path}: {str(e)}")

def compare_alpha_cls(src: Path, dst: Path, alpha: str) -> bool:
    """Compare alpha_cls.py files; return True if identical.

    Args:
        src (Path): Source directory.
        dst (Path): Destination directory.
        alpha (str): Alpha name.

    Returns:
        bool: True if files are identical, False otherwise.

    Raises:
        FileOperationError: If files cannot be compared.
    """
    src_cls = src / 'alpha_cls.py'
    dst_cls = dst / 'alpha_cls.py'
    if not src_cls.exists() or not dst_cls.exists():
        logger.warning(f"Missing alpha_cls.py: {src_cls} or {dst_cls}")
        return False
    try:
        # Preliminary check: compare file sizes and modification times
        if src_cls.stat().st_size != dst_cls.stat().st_size or \
           src_cls.stat().st_mtime != dst_cls.stat().st_mtime:
            return False
        return filecmp.cmp(src_cls, dst_cls, shallow=False)
    except Exception as e:
        raise FileOperationError(f"Error comparing alpha_cls.py for {alpha}: {str(e)}")

def move_alpha_directory(src: Path, dst: Path, alpha: str) -> None:
    """Move alpha directory to destination, overwriting if needed.

    Args:
        src (Path): Source directory.
        dst (Path): Destination directory.
        alpha (str): Alpha name.

    Raises:
        FileOperationError: If the move operation fails.
    """
    try:
        if dst.exists():
            backup_alpha(dst, alpha)
            rmtree_alpha_file(dst)
        shutil.move(src, dst)
        logger.info(f"Moved alpha {alpha} from {src} to {dst}")
    except Exception as e:
        raise FileOperationError(f"Failed to move alpha {alpha} from {src} to {dst}: {str(e)}")

def transfer_alpha(alpha: str, src: Path, dst: Path, nf_extension: str, out_decay: str) -> None:
    """Transfer single alpha to next stage, comparing alpha_cls.py if dst exists.

    Args:
        alpha (str): Alpha name.
        src (Path): Source directory.
        dst (Path): Destination directory.
        nf_extension (str): Next file extension.
        out_decay (str): Output decay path to remove.

    Raises:
        FileOperationError: If the transfer fails.
    """
    if dst.exists() and dst.is_dir() and not src.is_dir():
        logger.info(f"Skipping transfer for {alpha}: destination exists, source missing")
        return
    backup_alpha(dst, alpha)
    if not dst.exists():
        move_alpha_directory(src, dst, alpha)
        if nf_extension != "compare":
            decay_path = dst / out_decay
            rmtree_alpha_file
        fix_nested_path(dst, alpha)
    elif compare_alpha_cls(src, dst, alpha):
        logger.info(f"Alpha {alpha} at {src} is identical to destination at {dst}. No action taken.")
        rmtree_alpha_file(src)
    else:
        logger.info(f"Deleted source alpha {alpha} at {src} because it is identical to the destination at {dst}")
        move_alpha_directory(src, dst, alpha)
        fix_nested_path(dst, alpha)

def update_total_alpha(alpha: str, backtest_path: Path, update_alpha_path: Path, update_pnl_path: Path,
                      update_summary_path: Path, pnl_record_path: Path, summary_record_path: Path,
                      alpha_hdf_path: str, output_pnl_path: str, output_af_path: str) -> None:
    """Update files for total extension alpha.

    Args:
        alpha (str): Alpha name.
        backtest_path (Path): Backtest base path.
        update_alpha_path (Path): Path for updated alpha HDF5 files.
        update_pnl_path (Path): Path for updated PNL files.
        update_summary_path (Path): Path for updated summary files.
        pnl_record_path (Path): Path for PNL record files.
        summary_record_path (Path): Path for summary record files.
        alpha_hdf_path (str): Relative path for alpha HDF5 files.
        output_pnl_path (str): Relative path for PNL files.
        output_af_path (str): Relative path for summary files.

    Raises:
        FileOperationError: If any file operation fails.
    """
    if alpha in {x[:-31] for x in os.listdir(update_pnl_path) if len(x) >= 31}:
        logger.info(f"Skipping update for {alpha}: already exists in {update_pnl_path}")
        return

    paths = {
        'alpha': (backtest_path / 'signals_total' / alpha / alpha_hdf_path,
                  [update_alpha_path / f'{alpha}_alpha_zww.hdf5']),
        'pnl': (backtest_path / 'signals_total' / alpha / output_pnl_path,
                [update_pnl_path / f'{alpha}_daily_stats_info_stats_zww.csv',
                 pnl_record_path / f'{alpha}_daily_stats_info_stats_zww.csv']),
        'summary': (backtest_path / 'signals_total' / alpha / output_af_path,
                   [update_summary_path / f'{alpha}_summary_alpha_zww.csv',
                    summary_record_path / f'{alpha}_summary_alpha_zww.csv'])
    }

    for key, (src, dst_list) in paths.items():
        for dst in dst_list:
            try:
                if not dst.exists():
                    shutil.copy2(src, dst)
                    logger.info(f"Copied {src} to {dst}")
                else:
                    src_hash = hash_file(src)  # Compute hash for src once
                    dst_hash = hash_file(dst)  # Compute hash for dst once
                    if src_hash == dst_hash:  # Compare the stored hash values
                        if src.is_file():
                            os.remove(src)
                        elif src.is_dir():
                            shutil.rmtree(src)
                        logger.info(f"Deleted source file {src} (identical to {dst})")
                    else:
                        shutil.copy2(src, dst)
                        logger.info(f"Overwrote {dst} with {src} (contents differ)")
            except FileNotFoundError:
                logger.error(f"File not found: {src} or {dst}")
                continue
            except PermissionError:
                raise FileOperationError(f"Permission denied: {src} or {dst}")
            except Exception as e:
                raise FileOperationError(f"Error processing {src} to {dst}: {str(e)}")