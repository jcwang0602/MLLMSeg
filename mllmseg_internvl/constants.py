IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
QUAD_START_TOKEN = "<quad>"
QUAD_END_TOKEN = "</quad>"
REF_START_TOKEN = "<ref>"
REF_END_TOKEN = "</ref>"
BOX_START_TOKEN = "<box>"
BOX_END_TOKEN = "</box>"
SEG_TOKEN = "[SEG]"
REJ_TOKEN = "[REJ]"
DET_TOKEN = "[DET]"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.4814546, 0.4578275, 0.40821073)
CLIP_STD = (0.2686295, 0.2613025, 0.2757711)
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)

# LISA Questions and GSVA questions
CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

SHORT_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you segment the {class_name} in this image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please segment the {class_name} in this image.",
    DEFAULT_IMAGE_TOKEN + "\n" + "What is {class_name} in this image? Please respond with segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "\n" + "What is {class_name} in this image? Please output segmentation mask.",
]

SHORT_QUESTION_LIST_BBOX = [
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you detect the {class_name} in this image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please detect the {class_name} in this image.",
    DEFAULT_IMAGE_TOKEN + "\n" + "What is {class_name} in this image? Please respond with bounding box.",
    DEFAULT_IMAGE_TOKEN + "\n" + "What is {class_name} in this image? Please output bounding box.",
]
SHORT_QUESTION_LIST_MODE4 = [
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you segment {class_name} in this image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please segment {class_name} in this image.",
    DEFAULT_IMAGE_TOKEN + "\n" + "What are {class_name} in this image? Please respond with segmentation masks.",
    DEFAULT_IMAGE_TOKEN + "\n" + "What are {class_name} in this image? Please output segmentation masks.",
]

SHORT_QUESTION_LIST_MODE4_BBOX = [
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you detect {class_name} in this image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please detect {class_name} in this image.",
    DEFAULT_IMAGE_TOKEN + "\n" + "What are {class_name} in this image? Please respond with bounding box.",
    DEFAULT_IMAGE_TOKEN + "\n" + "What are {class_name} in this image? Please output bounding box.",
]

LONG_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please respond with segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please output segmentation mask.",
]

LONG_QUESTION_LIST_BBOX = [
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please respond with bounding box.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please output bounding box.",
]

EXPLANATORY_QUESTION_LIST = [
    "Please output segmentation mask and explain why.",
    "Please output segmentation mask and explain the reason.",
    "Please output segmentation mask and give some explaination.",
]

EXPLANATORY_QUESTION_LIST_BBOX = [
    "Please output bounding box and explain why.",
    "Please output bounding box and explain the reason.",
    "Please output bounding box and give some explaination.",
]

ANSWER_LIST = [
    "It is [SEG].",
    "Sure, [SEG].",
    "Sure, it is [SEG].",
    "Sure, the segmentation result is [SEG].",
    "[SEG].",
]

ANSWER_LIST_BBOX = [
    "It is [DET].",
    "Sure, [DET].",
    "Sure, it is [DET].",
    "Sure, the detection result is [DET].",
    "[DET].",
]

ANSWER_LIST_MODE1 = ["Here it is.", "Sure.", "Sure, this is the target.", "Sure, here is the segmentation result.", "Here you are."]

ANSWER_LIST_MODE4_START = ["The segmentation results are", "Sure, they are", "Sure,", "Sure,", "Sure,"]

ANSWER_LIST_MODE4_START_BBOX = ["The bounding box results are", "Sure, they are", "Sure,", "Sure,", "Sure,"]

ANSWER_LIST_MODE4_TEMPLATE = ["{class_name} [SEG]", "{class_name}:[SEG]", "the mask of {class_name} is [SEG]", "the segmentation of {class_name} is [SEG]", "the referred {class_name} is [SEG]"]

ANSWER_LIST_MODE4_TEMPLATE_BBOX = ["{class_name} [DET]", "{class_name}:[DET]", "the bounding box of {class_name} is [DET]", "the bounding box of {class_name} is [DET]", "the referred {class_name} is [DET]"]

ANSWER_LIST_MODE4_END = [".", ".", ".", ".", "."]

ANSWER_LIST_MODE4_END_BBOX = [".", ".", ".", ".", "."]
