# flake8: noqa
import re
import os
import sys

from ..smp import *

FAIL_MSG = "Failed to obtain answer via API."

logger = get_logger("ChartMimic")

# SET VLMEVAL_CHARTMIMIC_UTILS_PATH for chartmimic evaluator
# ".../VLMEvalKit/vlmeval..."
cur_path = os.path.abspath(__file__)
util_path = cur_path.replace("dataset/chartmimic.py", "dataset/utils/chartmimic")
os.environ["VLMEVAL_CHARTMIMIC_UTILS_PATH"] = util_path
if os.environ["VLMEVAL_CHARTMIMIC_UTILS_PATH"] not in sys.path:
    sys.path.insert(0, os.environ["VLMEVAL_CHARTMIMIC_UTILS_PATH"])

from .image_base import ImageBaseDataset
from .utils import build_judge, DEBUG_MESSAGE

# from ..utils import track_progress_rich
from ..dataset.utils.chartmimic.evaluator.text_evaluator import TextEvaluator
from ..dataset.utils.chartmimic.evaluator.chart_type_evaluator import ChartTypeEvaluator
from ..dataset.utils.chartmimic.evaluator.color_evaluator import ColorEvaluator
from ..dataset.utils.chartmimic.evaluator.layout_evaluator import LayoutEvaluator
from ..dataset.utils.chartmimic.mp_util import track_progress_rich_new

# from ..dataset.utils.chartmimic.evaluator.legend_evaluator import LegendEvaluator
# from ..dataset.utils.chartmimic.evaluator.grid_evaluator import GridEvaluator

judge_model = None
save_code_dir = None
sub_set_name = None
cur_work_dir = None
pdf_tmp_dir = None
# save_dir_name_map = {
#     "Direct Mimic": "direct",
#     "Customized Mimic": "customized",
# }

high_level_eval_prompt = {
    "instruction": "You are an excellent judge at evaluating visualization chart plots. The first image (reference image) is created using ground truth matplotlib code, and the second image (AI-generated image) is created using matplotlib code generated by an AI assistant. Your task is to score how well the AI-generated plot matches the ground truth plot.\n\n### Scoring Methodology:\nThe AI-generated image's score is based on the following criteria, totaling a score out of 100 points:\n\n1. **Chart Types (20 points)** Does the AI-generated image include all chart types present in the reference image (e.g., line charts, bar charts, etc.)?\n2. **Layout (10 points)** Does the arrangement of subplots in the AI-generated image match the reference image (e.g., number of rows and columns)?\n3. **Text Content (20 points)** Does the AI-generated image include all text from the reference image (e.g., titles, annotations, axis labels), excluding axis tick labels?\n4. **Data (20 points)** How accurately do the data trends in the AI-generated image resemble those in the original image and is the number of data groups the same as in the reference image?\n5. **Style (20 points)** Does the AI-generated image match the original in terms of colors (line colors, fill colors, etc.), marker types (point shapes, line styles, etc.), legends, grids, and other stylistic details?\n6. **Clarity (10 points)** Is the AI-generated image clear and free of overlapping elements?\n\n### Evaluation:\nCompare the two images head to head and provide a detailed assessment. Use the following format for your response:\n\n\n---\n\nComments:\n- Chart Types: ${your comment and subscore}\n- Layout: ${your comment and subscore}\n- Text Content: ${your comment and subscore}\n- Data: ${your comment and subscore}\n- Style: ${your comment and subscore}\n- Clarity: ${your comment and subscore}\n\nScore: ${your final score out of 100}\n\n---\n\nPlease use the above format to ensure the evaluation is clear and comprehensive.\n",
    "system_msg": "",
}


def image_path_to_data_uri(image_path):
    mime, _ = mimetypes.guess_type(image_path)
    if not mime:
        raise ValueError(f"Cannot determine MIME type for {image_path}")
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def run_once_with_images(pt, image_abs_path_list, retry=4):
    global judge_model
    # prefix = "data:image/jpeg;base64,"
    # img = prefix + image
    messages = [
        *[
            dict(type="image", value=image_path_to_data_uri(image_abs_path))
            for image_abs_path in image_abs_path_list
        ],
        dict(type="text", value=pt),
    ]
    ans = None
    while retry:
        try:
            ans = judge_model.generate(messages)
            return ans
        except Exception as e:
            logger.exception(f"Error in run_once_with_images: {e}")
            retry -= 1
    return ans


# def run_once_without_image(pt, retry=3):
#     global judge_model
#     messages = [
#         dict(type="text", value=pt),
#     ]
#     while retry:
#         try:
#             ans = judge_model.generate(messages)
#             return ans
#         except Exception as e:
#             logger.info(f"Error in run_once_without_image: {e}")
#             retry -= 1
#     return ans


# >>> util: extract python code from markdown text <<<
def extract_python_code(text):
    """Extract python code from markdown text."""
    code_matches = re.findall(r"```python(.*?)```", text, re.DOTALL)
    if not code_matches:
        return ""  # Return an empty string if no code block is found
    return code_matches[0]  # Return the first match


# >>> util: extract data code from
def get_variable_code(edit_ori_file):
    with open(edit_ori_file, "r") as f:
        code = f.read()
        pattern = re.compile(
            r"# ===================\n# Part 2: Data Preparation\n# ===================\n(.*?)# ===================\n# Part 3: Plot Configuration and Rendering\n# ===================",
            re.DOTALL,
        )
        match = pattern.search(code)

        if match:
            extracted_text = match.group(1)
            extracted_text = extracted_text.strip()
            extracted_text = (
                "#Variable Code Block\nimport warnings;warnings.filterwarnings('ignore', category=UserWarning);warnings.filterwarnings('ignore', category=FutureWarning);import matplotlib.pyplot as plt;import pandas as pd;import numpy as np;np.random.seed(0);import math;from matplotlib_venn import venn2;from matplotlib import cm;from scipy.stats import gaussian_kde;import networkx as nx;from matplotlib.gridspec import GridSpec;from scipy.stats import multivariate_normal;import colorsys;import matplotlib.colors as mcolors;from matplotlib.colors import LogNorm;from scipy.stats import norm;import matplotlib.gridspec as gridspec;import seaborn as sns\n"
                + extracted_text
            )
        else:
            print(edit_ori_file)
            raise ValueError("No match found")
    return extracted_text


# >>> util: clean escape characters in code string <<<
def clean_escape_chars(code: str) -> str:
    """
    Clean escape characters in code string to ensure proper execution.
    Handles common escape sequences and ensures proper string formatting.

    Args:
        code (str): The code string to clean

    Returns:
        str: Cleaned code string
    """
    # Common escape sequences to handle
    escape_map = {
        r"\\n": "\n",  # Newline
        r"\\r": "\r",  # Carriage return
        r"\\t": "\t",  # Tab
        r'\\"': '"',  # Double quote
        r"\\'": "'",  # Single quote
        r"\\\\": "\\",  # Backslash
        r"\\b": "\b",  # Backspace
        r"\\f": "\f",  # Form feed
        r"\\v": "\v",  # Vertical tab
    }

    # Replace escape sequences
    for escaped, unescaped in escape_map.items():
        code = code.replace(escaped, unescaped)

    return code


def _convert_single_page_pdf_to_png(pdf_path, output_path, dpi=350):
    from pdf2image import convert_from_path

    try:
        images = convert_from_path(pdf_path, dpi=dpi)
        images[0].save(output_path, "PNG")
    except Exception as e:
        logger.info(f"Error in converting pdf to image: {e}")
        return False
    return True


def extract_gpt_score(resp):
    # First match: standard or markdown-styled "Score: 91/100", "Score: **91/100**", etc
    pattern = r"^\s*Score:\s*[*_~`]*\**\s*(\d+)\s*/\s*100\s*[*_~`]*\**"
    m = re.search(pattern, resp, re.IGNORECASE | re.MULTILINE)
    if m:
        return int(m.group(1))

    # Fallback match: match "Score: 91", "Score: **91**", "Score: *91*", etc
    fallback_pattern = r"Score:\s*[*_~`]*\**\s*(\d+)\s*[*_~`]*\**"
    matches = list(re.finditer(fallback_pattern, resp, re.IGNORECASE))
    if matches:
        return int(matches[-1].group(1))

    return 0


def judge_one_item(item):
    score_dict = {}
    zero_score_dict = {
        "low_level": {
            "original_py_file": None,
            "generated_py_file": None,
            "text_metrics": {"precision": 0, "recall": 0, "f1": 0},
            "chart_type_metrics": {"precision": 0, "recall": 0, "f1": 0},
            "layout_metrics": {"precision": 0, "recall": 0, "f1": 0},
            "color_metrics": {"precision": 0, "recall": 0.0, "f1": 0},
        },
        "high_level": {
            "resp": None,
            "msg": None,
            "score": 0.0,
        },
    }

    global judge_model, save_code_dir, sub_set_name
    item = json.loads(item)
    # >>> 1. Run Code to Generate PY and PDF <<<
    # extract python code from item["prediction"]
    code = extract_python_code(item["prediction"])
    # clean code string: \\n -> \n...
    code = clean_escape_chars(code)
    # len(code) == 0 means no code generated or not format in ```python...```
    if len(code) == 0:
        logger.info(
            f"index: {item['index']}, no code extracted from prediction, return 0, zero_score_dict: {zero_score_dict}"
        )
        return 0, zero_score_dict

    # add data code to the beginning of code
    if "customized" in item["task"].lower():
        # extract data code from original file
        ground_truth_figure_code_file_rel = item["ground_truth_figure_code"]
        ROOT = LMUDataRoot()
        img_root = os.path.join(ROOT, "images", "ChartMimic")
        ground_truth_figure_code_file = os.path.join(
            img_root, ground_truth_figure_code_file_rel
        )
        data_code = get_variable_code(ground_truth_figure_code_file)
        # add data code to the beginning of code
        code = data_code + "\n" + code

    # save code to py and run to generate pdf
    # logger.info(f"save_code_dir: {save_code_dir}")
    if "direct" in item["task"].lower():
        save_dir_name = "direct"
    elif "customized" in item["task"].lower():
        save_dir_name = "customized"
    else:
        raise ValueError(f"Invalid task: {item['task']}")

    # save code to py and run to generate pdf
    output_py = (
        f"{save_code_dir}/ChartMimic/{sub_set_name}/{save_dir_name}/{item['index']}.py"
    )
    os.makedirs(os.path.dirname(output_py), exist_ok=True)
    # clean & add self redefined path
    code = re.sub(r"plt\.savefig\(.*\n*", "", code, flags=re.S)
    code = re.sub(r"plt.show\(.*\n*", "", code, flags=re.S)
    code = code.strip() + '\nplt.savefig("{}")'.format(output_py.replace(".py", ".pdf"))
    with open(output_py, "w") as f:
        f.write(code)
    # [Attention] run code with timeout, enhancement here
    # try generate pdf
    try:
        subprocess.run(
            ["python", output_py],
            timeout=120,
            capture_output=True,
            text=True,
        )
        logger.info(f"Successfully ran {output_py}")
    except subprocess.TimeoutExpired:
        logger.info(f"Timeout: Script {output_py} ran too long.")
    except Exception as e:
        # maybe could directly return 0, zero_score_dict
        logger.info(f"Error when running {output_py}: {e}")

    # check if pdf exists
    if not os.path.exists(output_py.replace(".py", ".pdf")):
        zero_score_dict["high_level"]["original_py_file"] = output_py
        logger.info(
            f"index: {item['index']}, run code failed, pdf does not exist, return 0, zero_score_dict: {zero_score_dict}"
        )
        return 0, zero_score_dict

    # try generate image (converted from pdf)
    if os.path.exists(output_py.replace(".py", ".pdf")):
        # if error when converting pdf to image, maybe could directly return 0, zero_score_dict
        _convert_single_page_pdf_to_png(
            output_py.replace(".py", ".pdf"), output_py.replace(".py", ".png")
        )
        # logger.info(f"converted pdf to image: {output_py.replace('.py', '.png')}")
        # breakpoint()

    # --- Got py and its pdf ---
    # >>> 2. Low Level Evaluation <<<
    text_evaluator = TextEvaluator(use_position=False, use_axs=False)
    chart_type_evaluator = ChartTypeEvaluator()
    color_evaluator = ColorEvaluator()
    layout_evaluator = LayoutEvaluator()
    # unused
    # legend_evaluator = LegendEvaluator(use_position=True)
    # grid_evaluator = GridEvaluator()

    ground_truth_figure_code_file_rel = item["ground_truth_figure_code"]
    ROOT = LMUDataRoot()
    img_root = os.path.join(ROOT, "images", "ChartMimic")
    ground_truth_figure_code_file = os.path.join(
        img_root, ground_truth_figure_code_file_rel
    )
    original_py_file = ground_truth_figure_code_file
    generated_py_file = output_py

    # logger.info(f"original_py_file: {original_py_file}")
    # logger.info(f"generated_py_file: {generated_py_file}")

    # global pdf_tmp_dir
    # os.chdir(pdf_tmp_dir)

    text_evaluator(
        generation_code_file=generated_py_file, golden_code_file=original_py_file
    )

    chart_type_evaluator(
        generation_code_file=generated_py_file, golden_code_file=original_py_file
    )

    color_evaluator(
        generation_code_file=generated_py_file, golden_code_file=original_py_file
    )

    layout_evaluator(
        generation_code_file=generated_py_file, golden_code_file=original_py_file
    )

    low_level_score_dict = {
        "original_py_file": original_py_file,
        "generated_py_file": generated_py_file,
        "text_metrics": text_evaluator.metrics,
        "chart_type_metrics": chart_type_evaluator.metrics,
        "layout_metrics": layout_evaluator.metrics,
        "color_metrics": color_evaluator.metrics,
    }

    score_dict["low_level"] = low_level_score_dict

    # >>> 3. High Level Evaluation <<<
    generated_pdf_image_file = generated_py_file.replace(".py", ".png")
    # check if generated_pdf_image_file exists
    if not os.path.exists(generated_pdf_image_file):
        # logger.info(f"Generated PDF image file {generated_pdf_image_file} does not exist")
        score_dict["high_level"] = {
            "resp": None,
            "msg": "Generated image file does not exist",
            "score": 0.0,
        }
        logger.info(f"index: {item['index']}, return 0, score_dict: {score_dict}")
        return 0, score_dict

    # image order should align with prompt
    resp = run_once_with_images(
        high_level_eval_prompt["instruction"],
        [original_py_file.replace(".py", ".png"), generated_pdf_image_file],
    )
    if resp is None:
        logger.error("Error in getting response from judge model!")
        score_dict["high_level"] = {
            "resp": None,
            "msg": "Error in getting response from judge model!",
            "score": 0.0,
        }
        logger.info(f"index: {item['index']}, return -1, score_dict: {score_dict}")
        return -1, score_dict
    else:
        # logger.info(f"Successfully got response from judge model:\n{resp}")
        score_dict["high_level"] = {
            "resp": resp,
            "msg": "Successfully got response from judge model!",
            "score": extract_gpt_score(resp),
        }
        logger.info(f"index: {item['index']}, return 0, score_dict: {score_dict}")
    return 0, score_dict


class ChartMimic(ImageBaseDataset):
    TYPE = "VQA"

    # TODO: add dataset url and md5
    DATASET_URL = {
        "ChartMimic_v1_customized": "https://opencompass.openxlab.space/utils/VLMEval/ChartMimic_v1_customized.tsv",
        "ChartMimic_v1_direct": "https://opencompass.openxlab.space/utils/VLMEval/ChartMimic_v1_direct.tsv",
        # v2
        "ChartMimic_v2_customized": "https://opencompass.openxlab.space/utils/VLMEval/ChartMimic_v2_customized.tsv",
        "ChartMimic_v2_customized_600": "https://opencompass.openxlab.space/utils/VLMEval/ChartMimic_v2_customized_600.tsv",
        "ChartMimic_v2_customized_1800": "https://opencompass.openxlab.space/utils/VLMEval/ChartMimic_v2_customized_1800.tsv",
        "ChartMimic_v2_direct": "https://opencompass.openxlab.space/utils/VLMEval/ChartMimic_v2_direct.tsv",
        "ChartMimic_v2_direct_600": "https://opencompass.openxlab.space/utils/VLMEval/ChartMimic_v2_direct_600.tsv",
        "ChartMimic_v2_direct_1800": "https://opencompass.openxlab.space/utils/VLMEval/ChartMimic_v2_direct_1800.tsv"
    }
    DATASET_MD5 = {
        "ChartMimic_v1_customized": "d636eca077e75e39fd2600889bf0284e",
        "ChartMimic_v1_direct": "d0fb410970cab0c666bbacf7d9f0cfb3",
        "ChartMimic_v2_customized": "390e715dbdfbad3ff788fffa91945405",
        "ChartMimic_v2_customized_600": "79907e8f9edc5e0eccbbcfc9cbe8a235",
        "ChartMimic_v2_customized_1800": "a6cf57807c07d328689872a77c9f847a",
        "ChartMimic_v2_direct": "1c8b444bd681f808f77f06037866eb19",
        "ChartMimic_v2_direct_600": "3d8d8afecccb6e8feacbcec6834f45f5",
        "ChartMimic_v2_direct_1800": "340331019c7eaa56cc02080656b66c3c"
    }

    def dump_image(self, line):
        input_figure_path_rel = line["input_figure"]
        ROOT = LMUDataRoot()
        img_root = os.path.join(ROOT, 'images', 'ChartMimic')
        input_figure_path = os.path.join(img_root, input_figure_path_rel)
        tgt_path = [input_figure_path]
        return tgt_path

    def prepare_tsv(self, url, file_md5=None):
        data_root = LMUDataRoot()
        os.makedirs(data_root, exist_ok=True)
        update_flag = False
        file_name = url.split("/")[-1]
        data_path = osp.join(data_root, file_name)
        self.data_path = data_path
        if osp.exists(data_path) and (file_md5 is None or md5(data_path) == file_md5):
            pass
        else:
            warnings.warn("The dataset tsv is not downloaded")
            download_file(url, data_path)
            update_flag = True

        if file_size(data_path, "GB") > 1:
            local_path = data_path.replace(".tsv", "_local.tsv")
            if (
                not osp.exists(local_path)
                or os.environ.get("FORCE_LOCAL", None)
                or update_flag
            ):
                from ..tools import LOCALIZE

                LOCALIZE(data_path, local_path)
            data_path = local_path
        # Extra check for images
        py_root = os.path.join(LMUDataRoot(), "images", "ChartMimic")
        v1_path = osp.join(py_root, "v1")
        # Check if py_root/v1 exists
        if not osp.exists(os.path.join(v1_path, "customized_500")) or not osp.exists(
            os.path.join(v1_path, "ori_500")
        ):
            # Download v1
            warnings.warn("Python files v1 needed by ChartMimic are not downloaded")
            os.makedirs(v1_path, exist_ok=True)
            v1_tar = osp.join(v1_path, "v1.tar.gz")
            if not osp.exists(v1_tar):
                print("Downloading ChartMimic v1 files...")
                subprocess.run(
                    [
                        "wget",
                        "https://hf-mirror.com/datasets/ChartMimic/ChartMimic/resolve/main/dataset-old.tar.gz",
                        "-O",
                        v1_tar,
                    ],
                    check=True,
                )
            print("Extracting v1...")
            # subprocess.run([
            #     "tar", "-xzvf", v1_tar, "-C", v1_path
            # ], check=True)
            try:
                subprocess.run(
                    ["tar", "-xzvf", v1_tar, "--no-same-owner", "-C", v1_path],
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                warnings.warn(f"tar extract v1 warning, try to continue. error: {e}")
        v2_path = osp.join(py_root, "v2")
        if (
            not osp.exists(os.path.join(v2_path, "customized_1800"))
            or not osp.exists(os.path.join(v2_path, "direct_1800"))
            or not osp.exists(os.path.join(v2_path, "customized_600"))
            or not osp.exists(os.path.join(v2_path, "direct_600"))
        ):
            warnings.warn("Python files v2 needed by ChartMimic are not downloaded")
            os.makedirs(v2_path, exist_ok=True)
            v2_tar = osp.join(v2_path, "v2.tar.gz")
            if not osp.exists(v2_tar):
                print("Downloading ChartMimic v2 files...")
                subprocess.run(
                    [
                        "wget",
                        "https://hf-mirror.com/datasets/ChartMimic/ChartMimic/resolve/main/dataset-iclr.tar.gz",
                        "-O",
                        v2_tar,
                    ],
                    check=True,
                )
            print("Extracting v2...")
            try:
                subprocess.run(
                    ["tar", "-xzvf", v2_tar, "--no-same-owner", "-C", v2_path],
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                warnings.warn(f"tar extract v2 warning, try to continue. error: {e}")

        return load(data_path)

    # Given one data record, return the built prompt (a multi-modal message), can override
    # Actually, all lines have single image
    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        # no "image" in tsv, so self.meta_only is True
        # logger.info(f"self.meta_only: {self.meta_only}")
        # logger.info(line.keys())

        input_figure_path_rel = line["input_figure"]
        instruction = line["question"]

        ROOT = LMUDataRoot()
        img_root = os.path.join(ROOT, "images", "ChartMimic")
        input_figure_path = os.path.join(img_root, input_figure_path_rel)

        msgs = []
        msgs = [dict(type="image", value=input_figure_path)]
        msgs = [dict(type="text", value=instruction)] + msgs

        return msgs


    def evaluate(self, eval_file, **judge_kwargs):
        def judge_one_item_success(item):
            return item["high_level"]["resp"] not in [FAIL_MSG, "", None, "null", "None"] \
                and item["high_level"]["msg"] not in ["Generated image file does not exist", ""]

        # Test dependencies first
        try:
            from pdf2image import convert_from_path
            from colormath.color_objects import sRGBColor, LabColor
            import squarify
            import matplotlib_venn
            import PIL
        except ImportError as e:
            logging.critical(
                "Please follow the requirements (see vlmeval/dataset/utils/chartmimic/eval_req.txt) \
                             to install dependency package for chartmimic evaluation."
            )
            raise e
        
        # Test pdf2image functionality by creating a simple test PDF
        example_pdf_path = os.path.join(LMUDataRoot(), "chartmimic_test.pdf")
        output_png_path = os.path.join(LMUDataRoot(), "chartmimic_test.png")
        
        try:
            # Create a simple test PDF using matplotlib
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1, figsize=(4, 3))
            ax.plot([1, 2, 3], [1, 4, 2])
            ax.set_title("Test Chart")
            plt.savefig(example_pdf_path, format='pdf')
            plt.close()
            
            # Test pdf2image conversion
            images = convert_from_path(example_pdf_path, dpi=350)
            images[0].save(output_png_path, "PNG")
            logger.info("Successfully tested pdf2image functionality with generated test PDF")
        except Exception as e:
            logging.critical(
                "Please install poppler-utils in your system (e.g. sudo apt-get install poppler-utils)."
            )
            raise e
        finally:
            # Clean up test files
            if os.path.exists(example_pdf_path):
                os.remove(example_pdf_path)
            if os.path.exists(output_png_path):
                os.remove(output_png_path)

        infer_data_all = load(eval_file).to_dict(orient="records")

        suffix = eval_file.split(".")[-1]
        print(f"judge_kwargs: {judge_kwargs}")
        infer_model = judge_kwargs["model"]
        storage = os.path.abspath(
            eval_file.replace(f".{suffix}", f"_{infer_model}.jsonl")
        )
        score_file = os.path.abspath(
            eval_file.replace(f".{suffix}", f"_{infer_model}_score.csv")
        )
        # use abs path because of using os.chdir()
        tmp_file = os.path.abspath(
            eval_file.replace(f".{suffix}", f"_{infer_model}_tmp.pkl")
        )
        # actually the --api-nproc
        nproc = judge_kwargs.pop("nproc", 8)
        logger.info(f"nproc: {nproc}")
        global save_code_dir, sub_set_name
        # [Attention] should use absolute dir here
        eval_file_abs_path = os.path.abspath(eval_file)
        save_code_dir = os.path.dirname(eval_file_abs_path)

        # dataset_name is subset name like ChartMimic_1000
        sub_set_name = self.dataset_name

        # params prepare for track_progress_rich
        params_all = [json.dumps(item) for item in infer_data_all]
        indices_all = [line["index"] for line in infer_data_all]

        ans = {}
        if os.path.exists(tmp_file):
            tmp_data = load(tmp_file)
            for k, v in tmp_data.items():
                # -1 means error for getting response from judge model, so try to rejudge for this item
                if v[0] == 0 and judge_one_item_success(v[1]):
                    ans[k] = v
            logger.info(f"Tmp file exists, loaded {len(ans)} data from {tmp_file}")

        tups = [x for x, i in zip(params_all, indices_all) if i not in ans]
        indices = [i for i in indices_all if i not in ans]

        # save current work dir
        global cur_work_dir, pdf_tmp_dir
        cur_work_dir = os.getcwd()
        pdf_tmp_dir = os.path.join(save_code_dir, "chart_mimic_tmp", f"{sub_set_name}")
        os.makedirs(pdf_tmp_dir, exist_ok=True)
        os.chdir(pdf_tmp_dir)

        # >>> judge <<<
        if len(indices):
            # judge_kwargs['system_prompt'] = SYSTEM_PROMPT
            judge_kwargs["temperature"] = 0
            judge_kwargs["img_detail"] = "high"
            judge_kwargs["timeout"] = 100
            global judge_model
            judge_model = build_judge(max_tokens=1024, **judge_kwargs)

            assert judge_model.working(), (
                "ChartMimic evaluation requires a working OPENAI API\n" + DEBUG_MESSAGE
            )

            # if len(indices):
            new_results = track_progress_rich_new(
                judge_one_item,
                tups,
                nproc=nproc,
                keys=indices,
                save=tmp_file,
            )
            for k, v in zip(indices, new_results):
                ans[k] = v

        for item in infer_data_all:
            # ans[i] is a tuple, (0 / -1, score_dict), only use score_dict
            item["judge_result"] = ans[item["index"]][1]

        # storage is a jsonl file
        with open(storage, "w") as f:
            for item in infer_data_all:
                f.write(json.dumps(item) + "\n")

        # judge finished, rm tmp dir
        os.chdir(cur_work_dir)
        if os.path.exists(pdf_tmp_dir):
            shutil.rmtree(pdf_tmp_dir)
        # breakpoint()

        # logger.info(f"storage: {storage}")
        eval_data_all = load(storage)
        # result_df = pd.DataFrame(columns=["example_count", "exec_rate", "text_score","layout_score",  "type_score",  "color_score", "average", f"gpt_score({judge_kwargs['model']})", "overall"])

        # filter out items that do not have judge_result["low_level"] and judge_result["high_level"]: failed item need rejudge
        old_len = len(eval_data_all)
        eval_data_all = [
            item
            for item in eval_data_all
            if "judge_result" in item
            and "low_level" in item["judge_result"]
            and "high_level" in item["judge_result"]
        ]
        new_len = len(eval_data_all)
        logger.info(f"filter out {old_len - new_len} items for no judge_result in item")

        old_len = len(eval_data_all)
        eval_data_all = [
            item
            for item in eval_data_all
            # if judge_one_item_success(item["judge_result"])
        ]
        new_len = len(eval_data_all)
        logger.info(
            f"filter out {old_len - new_len} items for FAIL_MSG in high_level resp"
        )

        def compute_metrics(eval_data):
            result = {
                "example_count": len(eval_data),
            }

            denominator = len(eval_data)
            if denominator == 0:
                # Avoid division by zero, return zeros
                result.update(
                    {
                        "exec_rate": 0,
                        "text_score": 0,
                        "layout_score": 0,
                        "chart_type_score": 0,
                        "color_score": 0,
                        "average": 0,
                        "gpt_score": 0,
                        "overall": 0,
                    }
                )
                return result

            pdf_file_cnt = 0
            text_score_sum = 0
            layout_score_sum = 0
            type_score_sum = 0
            color_score_sum = 0
            gpt_score_sum = 0

            for item in eval_data:
                py_file = item["judge_result"]["low_level"]["generated_py_file"]
                if py_file and os.path.exists(py_file.replace(".py", ".pdf")):
                    pdf_file_cnt += 1

                text_score_sum += item["judge_result"]["low_level"]["text_metrics"][
                    "f1"
                ]
                layout_score_sum += item["judge_result"]["low_level"]["layout_metrics"][
                    "f1"
                ]
                type_score_sum += item["judge_result"]["low_level"][
                    "chart_type_metrics"
                ]["f1"]
                color_score_sum += item["judge_result"]["low_level"]["color_metrics"][
                    "f1"
                ]
                gpt_score_sum += item["judge_result"]["high_level"]["score"]

            result["exec_rate"] = pdf_file_cnt / denominator * 100
            result["text_score"] = text_score_sum / denominator * 100
            result["layout_score"] = layout_score_sum / denominator * 100
            result["chart_type_score"] = type_score_sum / denominator * 100
            result["color_score"] = color_score_sum / denominator * 100
            result["average"] = (
                result["text_score"]
                + result["layout_score"]
                + result["chart_type_score"]
                + result["color_score"]
            ) / 4
            result["gpt_score"] = gpt_score_sum / denominator
            result["overall"] = (result["average"] + result["gpt_score"]) / 2

            return result

        # Collect unique task values
        task_values = sorted(
            set(
                item.get("task")
                for item in eval_data_all
                if item.get("task") is not None
            )
        )

        # Create splits dict
        splits = {
            "all": eval_data_all,
            **{
                task: [item for item in eval_data_all if item.get("task") == task]
                for task in task_values
            },
        }

        all_results = []
        for split_name, data in splits.items():
            result = compute_metrics(data)
            result["split"] = split_name
            all_results.append(result)

        score_df = pd.DataFrame(all_results)
        # reorder columns
        cols = ["split"] + [col for col in score_df.columns if col != "split"]
        score_df = score_df[cols]
        dump(score_df, score_file)
        return score_df
