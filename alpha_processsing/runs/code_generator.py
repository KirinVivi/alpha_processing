import ast
from typing import List
from pathlib import Path
from utils import data_utils as du
import path_set
import logging

logger = logging.getLogger(__name__)


def _parse_function_string(func_str: str) -> List[str]:
    """
    Safely parse a string representation of a list into a Python list.
    Example: "['rank(10)', 'ts_max']" -> ['rank(10)', 'ts_max']
    """
    if not func_str or func_str == 'None':
        return []
    try:
        # ast.literal_eval is a safe way to evaluate a string containing a Python literal
        parsed_list = ast.literal_eval(func_str)
        if isinstance(parsed_list, list):
            return parsed_list
        else:
            # Handle cases where the string is valid but not a list, e.g., "'ts_rank(10)'"
            return [str(parsed_list)]
    except (ValueError, SyntaxError) as e:
        logger.error(f"Failed to parse function string: {func_str}. Error: {e}")
        # Depending on desired behavior, you might want to raise the exception
        # raise ValueError(f"Invalid function string format: {func_str}") from e
        return []

def _generate_processing_code(functions: List[str]) -> str:
    """
    Generates lines of Python code from a list of function strings.
    """
    code_lines = []
    for func in functions:
        # Assumes utils.extract_values_from_string is a pre-existing helper
        # that returns (func_name, params_str) or None
        params = du.extract_values_from_string(func)
        if params:
            func_name, func_args = params
            code_lines.append(f"    alpha = dp.utils2.{func_name}(alpha, {func_args})\n")
        else:
            code_lines.append(f"    alpha = dp.utils2.{func}(alpha)\n")
    return "".join(code_lines)

def generate_alpha_code(
    cal_data_prefix: str,
    fill_func_str: str,
    cal_funcs_str: str,
    smooth_funcs_str: str,
    alpha_txt: str
) -> None:
    """
    Generate alpha_cls.py and alpha_config.py using a modular approach.

    Args:
        cal_data_prefix: Prefix for the data source.
        fill_func_str: A single function name for initial data filling (or 'None').
        cal_funcs_str: A string representation of a list of calculation functions.
                       Example: "['rank(10)', 'decay_linear(5)']"
        smooth_funcs_str: A string representation of a list of smoothing functions.
        alpha_txt: Text to be injected into the alpha template.
    """
    alpha_name = f"{cal_data_prefix}_{fill_func_str}_{cal_funcs_str}_{smooth_funcs_str}".replace('__', '_').replace('[','').replace(']','').replace("'",'').replace(", ", "_")
    output_path = Path(path_set.backtest_path) / 'signals_106' / alpha_name
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Generate the data processing code block
    fill_code = f"alpha = dp.utils2.{fill_func_str}(data)\n" if fill_func_str and fill_func_str != 'None' else "alpha = data.copy()\n"
    cal_funcs_list = _parse_function_string(cal_funcs_str)
    smooth_funcs_list = _parse_function_string(smooth_funcs_str)
    
    cal_code = _generate_processing_code(cal_funcs_list)
    smooth_code = _generate_processing_code(smooth_funcs_list)
    
    work_func_txt = fill_code + cal_code + smooth_code

    # 2. Process and write the alpha class file
    template_path = Path(path_set.code_recorded_path) / 'alpha_cls_txt' / f'{cal_data_prefix}.txt'
    try:
        template_content = template_path.read_text(encoding='utf-8')
        final_code = template_content.replace('{handle_func}', work_func_txt).replace('{alpha_part}', alpha_txt)
        (output_path / 'alpha_cls.py').write_text(final_code, encoding='utf-8')
    except FileNotFoundError:
        raise FileNotFoundError(f"Alpha class template not found: {template_path}")

    # 3. Process and write the config file
    config_template_path = Path(path_set.demo_path) / 'alpha_config.txt'
    try:
        config_content = config_template_path.read_text(encoding='utf-8')
        (output_path / 'alpha_config.py').write_text(config_content, encoding='utf-8')
    except FileNotFoundError:
        raise FileNotFoundError(f"Config template not found: {config_template_path}")

    # 4. Create the __init__.py file
    (output_path / '__init__.py').touch()
    
    logger.info(f"Successfully generated alpha code at: {output_path}")