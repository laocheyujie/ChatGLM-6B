"""Tokenization classes for ChatGLM."""
from typing import List, Optional, Union
import os

from transformers.tokenization_utils import PreTrainedTokenizer  # 从 transformers 包导入预训练的词条化工具类
from transformers.utils import logging, PaddingStrategy  # 导入 transformers 的日志和填充策略工具类
from transformers.tokenization_utils_base import EncodedInput, BatchEncoding  # 导入词条化相关工具类
from typing import Dict  # 导入字典类型
import sentencepiece as spm  # 导入 sentencepiece，一个开源的词条化工具
import numpy as np  # 导入 numpy，用于科学计算


logger = logging.get_logger(__name__)

# 定义预训练位置嵌入大小的常量
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "THUDM/chatglm-6b": 2048,
}


class TextTokenizer:
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()  # 创建一个 SentencePieceProcessor 实例
        self.sp.Load(model_path)  # 加载模型
        self.num_tokens = self.sp.vocab_size()  # 获取模型的词汇表大小

    def encode(self, text):
        return self.sp.EncodeAsIds(text)  # 将文本编码为ID序列

    def decode(self, ids: List[int]):
        return self.sp.DecodeIds(ids)  # 将ID序列解码为文本

    def tokenize(self, text):
        return self.sp.EncodeAsPieces(text)  # 将文本分割为词条序列

    def convert_tokens_to_string(self, tokens):
        return self.sp.DecodePieces(tokens)  # 将词条序列解码为文本

    def convert_tokens_to_ids(self, tokens):
        return [self.sp.PieceToId(token) for token in tokens]  # 将词条序列转换为ID序列

    def convert_token_to_id(self, token):
        return self.sp.PieceToId(token)  # 将单个词条转换为ID

    def convert_id_to_token(self, idx):
        return self.sp.IdToPiece(idx)  # 将ID转换为词条

    def __len__(self):
        return self.num_tokens  # 返回词汇表大小


class SPTokenizer:
    def __init__(
            self,
            vocab_file,
            num_image_tokens=20000,
            max_blank_length=80,
            byte_fallback=True,
    ):
        assert vocab_file is not None  # 检查词汇表文件是否存在
        self.vocab_file = vocab_file  # 保存词汇表文件路径
        self.num_image_tokens = num_image_tokens  # 保存图像词条数量
        self.special_tokens = ["[MASK]", "[gMASK]", "[sMASK]", "<unused_0>", "<sop>", "<eop>", "<ENC>", "<dBLOCK>"]  # 定义特殊词条
        self.max_blank_length = max_blank_length  # 定义最大空白长度
        self.byte_fallback = byte_fallback  # 设置字节回退标记
        self.text_tokenizer = TextTokenizer(vocab_file)  # 创建文本词条化工具

    def _get_text_tokenizer(self):
        return self.text_tokenizer  # 获取文本词条化工具

    @staticmethod
    def get_blank_token(length: int):
        assert length >= 2
        return f"<|blank_{length}|>"  # 获取空白词条

    @staticmethod
    def get_tab_token():
        return f"<|tab|>"  # 获取制表符词条

    @property
    def num_text_tokens(self):
        return self.text_tokenizer.num_tokens  # 获取文本词条数量

    @property
    def num_tokens(self):
        return self.num_image_tokens + self.num_text_tokens  # 获取总词条数量

    @staticmethod
    def _encode_whitespaces(text: str, max_len: int = 80):
        text = text.replace("\t", SPTokenizer.get_tab_token())  # 替换制表符
        for i in range(max_len, 1, -1):
            text = text.replace(" " * i, SPTokenizer.get_blank_token(i))  # 替换多个连续空格
        return text

    def _preprocess(self, text: str, linebreak=True, whitespaces=True):
        if linebreak:
            text = text.replace("\n", "<n>")  # 替换换行符
        if whitespaces:
            text = self._encode_whitespaces(text, max_len=self.max_blank_length)  # 编码空白字符
        return text

    def encode(
            self, text: str, linebreak=True, whitespaces=True, add_dummy_prefix=True
    ) -> List[int]:
        """
        文本编码方法
        @param text: Text to encode.
        @param linebreak: Whether to encode newline (\n) in text.
        @param whitespaces: Whether to encode multiple whitespaces or tab in text, useful for source code encoding.
        @param special_tokens: Whether to encode special token ([MASK], [gMASK], etc.) in text.
        @param add_dummy_prefix: Whether to add dummy blank space in the beginning.
        """
        text = self._preprocess(text, linebreak, whitespaces)  # 预处理文本
        if not add_dummy_prefix:
            text = "<n>" + text
        tmp = self._get_text_tokenizer().encode(text)  # 编码文本
        tokens = [x + self.num_image_tokens for x in tmp]  # 将文本词条ID转换为包含图像词条ID的序列
        return tokens if add_dummy_prefix else tokens[2:]

    def postprocess(self, text):
        text = text.replace("<n>", "\n")  # 替换换行词条
        text = text.replace(SPTokenizer.get_tab_token(), "\t")  # 替换制表符词条
        for i in range(2, self.max_blank_length + 1):
            text = text.replace(self.get_blank_token(i), " " * i)  # 替换空白词条
        return text

    def decode(self, text_ids: List[int]) -> str:
        ids = [int(_id) - self.num_image_tokens for _id in text_ids]  # 将包含图像词条的ID序列转换为文本词条ID序列
        ids = [_id for _id in ids if _id >= 0]  # 删除非文本词条ID
        text = self._get_text_tokenizer().decode(ids)  # 解码ID序列为文本
        text = self.postprocess(text)  # 对文本进行后处理
        return text

    def decode_tokens(self, tokens: List[str]) -> str:
        text = self._get_text_tokenizer().convert_tokens_to_string(tokens)  # 将词条序列解码为文本
        text = self.postprocess(text)  # 对文本进行后处理
        return text

    def tokenize(
            self, text: str, linebreak=True, whitespaces=True, add_dummy_prefix=True
    ) -> List[str]:
        """
        文本分词方法
        @param text: Text to encode.
        @param linebreak: Whether to encode newline (\n) in text.
        @param whitespaces: Whether to encode multiple whitespaces or tab in text, useful for source code encoding.
        @param special_tokens: Whether to encode special token ([MASK], [gMASK], etc.) in text.
        @param add_dummy_prefix: Whether to add dummy blank space in the beginning.
        """
        text = self._preprocess(text, linebreak, whitespaces)  # 预处理文本
        if not add_dummy_prefix:
            text = "<n>" + text
        tokens = self._get_text_tokenizer().tokenize(text)  # 分词
        return tokens if add_dummy_prefix else tokens[2:]

    def __getitem__(self, x: Union[int, str]):
        if isinstance(x, int):
            if x < self.num_image_tokens:
                return "<image_{}>".format(x)  # 如果是图像词条，返回词条形式
            else:
                return self.text_tokenizer.convert_id_to_token(x - self.num_image_tokens)  # 如果是文本词条，返回文本词条
        elif isinstance(x, str):
            if x.startswith("<image_") and x.endswith(">") and x[7: -1].isdigit():
                return int(x[7: -1])  # 如果是图像词条形式，返回词条ID
            else:
                return self.text_tokenizer.convert_token_to_id(x) + self.num_image_tokens  # 如果是文本词条，返回包含图像词条的ID
        else:
            raise ValueError("The key should be str or int.")  # 如果不是整数或字符串，抛出异常


class ChatGLMTokenizer(PreTrainedTokenizer):
    """
    基于PreTrainedTokenizer定义一个新的分词器类
    Construct a ChatGLM tokenizer. Based on byte-level Byte-Pair-Encoding.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
    """

    vocab_files_names = {"vocab_file": "ice_text.model"}  # 设定词汇表文件名称
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES  # 预设模型输入的最大尺寸
    model_input_names = ["input_ids", "attention_mask", "position_ids"]  # 预设模型输入的名称列表

    def __init__(                    # 定义初始化函数
            self,
            vocab_file,              # 词汇表文件路径
            do_lower_case=False,     # 是否对文本做小写转换
            remove_space=False,      # 是否移除文本中的空格
            bos_token='<sop>',       # 文本开头的特殊词条
            eos_token='<eop>',       # 文本结尾的特殊词条
            end_token='</s>',        # 文本结束的特殊词条
            mask_token='[MASK]',     # 遮蔽词条
            gmask_token='[gMASK]',   # gMASK词条
            padding_side="left",     # 填充侧（左侧填充或右侧填充）
            pad_token="<pad>",       # 填充词条
            unk_token="<unk>",       # 未知词条
            num_image_tokens=20000,  # 图像词条的数量
            **kwargs                 # 其他参数
    ) -> None:
        super().__init__(            # 调用父类的初始化函数
            do_lower_case=do_lower_case,
            remove_space=remove_space,
            padding_side=padding_side,
            bos_token=bos_token,
            eos_token=eos_token,
            end_token=end_token,
            mask_token=mask_token,
            gmask_token=gmask_token,
            pad_token=pad_token,
            unk_token=unk_token,
            num_image_tokens=num_image_tokens,
            **kwargs
        )

        self.do_lower_case = do_lower_case  # 是否进行小写转换
        self.remove_space = remove_space  # 是否移除空格
        self.vocab_file = vocab_file  # 词汇表文件
 
        self.bos_token = bos_token  # 文本开头的特殊词条
        self.eos_token = eos_token  # 文本结尾的特殊词条
        self.end_token = end_token  # 文本结束的特殊词条
        self.mask_token = mask_token  # 遮蔽词条
        self.gmask_token = gmask_token  # gMASK词条
 
        self.sp_tokenizer = SPTokenizer(vocab_file, num_image_tokens=num_image_tokens)  # 初始化SPTokenizer

        """ Initialisation """
    # 以下部分是定义了一些属性和方法
    @property
    def gmask_token_id(self) -> Optional[int]:  # 获取gmask词条的id
        if self.gmask_token is None:  # 若不存在，则返回None
            return None
        return self.convert_tokens_to_ids(self.gmask_token)  # 返回gmask词条对应的id

    @property
    def end_token_id(self) -> Optional[int]:  # 获取end词条的id
        """
        `Optional[int]`: Id of the end of context token in the vocabulary. Returns `None` if the token has not been
        set.
        """
        if self.end_token is None:  # 若不存在，则返回None
            return None
        return self.convert_tokens_to_ids(self.end_token)  # 返回end词条对应的id

    @property
    def vocab_size(self):  # 获取词汇表的大小
        """ Returns vocab size """
        return self.sp_tokenizer.num_tokens  # 返回词汇表的大小

    def get_vocab(self):  # 获取词汇表
        """ Returns vocab as a dict """
        vocab = {self._convert_id_to_token(i): i for i in range(self.vocab_size)}
        # 这行代码将任何额外添加的词汇（可能是之后添加到模型中的）合并到主词汇表中
        vocab.update(self.added_tokens_encoder)  # self.added_tokens_encoder是一个字典，其中包含额外的词汇项及其对应的ID
        return vocab  # 返回词汇表
 
    def preprocess_text(self, inputs):  # 文本预处理函数
        if self.remove_space:  # 若需要移除空格
            outputs = " ".join(inputs.strip().split())  # 则移除多余的空格
        else:
            outputs = inputs  # 否则保持不变

        if self.do_lower_case:  # 若需要进行小写转换
            outputs = outputs.lower()  # 则转换为小写
 
        return outputs  # 返回预处理后的文本

    def _tokenize(self, text, **kwargs):  # 分词函数
        """ Returns a tokenized string. """
        text = self.preprocess_text(text)  # 对文本进行预处理

        seq = self.sp_tokenizer.tokenize(text)  # 对文本进行分词

        return seq  # 返回分词结果

    def convert_tokens_to_string(self, tokens: List[str]) -> str:  # 将词条转化为字符串
        return self.sp_tokenizer.decode_tokens(tokens)  # 解码词条
 
    def _decode(
            self,
            token_ids: Union[int, List[int]],
            **kwargs
    ) -> str:
        # 对id进行解码
        if isinstance(token_ids, int):  # 如果输入是单个id
            token_ids = [token_ids]  # 则将其转化为列表
        if len(token_ids) == 0:  # 如果输入为空
            return ""  # 则返回空字符串
        if self.pad_token_id in token_ids:  # 如果填充id在输入中 remove pad  
            token_ids = list(filter((self.pad_token_id).__ne__, token_ids))  # 则移除填充id
        return super()._decode(token_ids, **kwargs)  # 返回父类的解码函数

    def _convert_token_to_id(self, token):  # 将词条转化为id
        """ Converts a token (str) in an id using the vocab. """
        return self.sp_tokenizer[token]  # 使用sp_tokenizer进行转换

    def _convert_id_to_token(self, index):  # 将id转化为词条
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.sp_tokenizer[index]  # 使用sp_tokenizer进行转换

    def save_vocabulary(self, save_directory, filename_prefix=None):  # 保存词汇表到指定目录
        """
        Save the vocabulary and special tokens file to a directory.

        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.
            filename_prefix (`str`, *optional*):
                An optional prefix to add to the named of the saved files.

        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        # 将词汇表及特殊词条文件保存到目录
        if os.path.isdir(save_directory):  # 如果保存目录存在
            vocab_file = os.path.join(
                save_directory, self.vocab_files_names["vocab_file"]
            )  # 则构建vocab文件路径
        else:
            vocab_file = save_directory  # 否则vocab文件就是保存目录

        with open(self.vocab_file, 'rb') as fin:  # 打开vocab文件
            proto_str = fin.read()  # 读取文件内容

        with open(vocab_file, "wb") as writer:  # 打开待写入的文件
            writer.write(proto_str)  # 写入内容

        return (vocab_file,)  # 返回保存的文件路径

    # 以下是与特殊词条有关的方法
    def build_inputs_with_special_tokens(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A BERT sequence has the following format:

        - single sequence: `[CLS] X [SEP]`
        - pair of sequences: `[CLS] A [SEP] B [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        # 构建带有特殊词条的输入
        gmask_id = self.sp_tokenizer[self.gmask_token]  # 获取gmask的id
        eos_id = self.sp_tokenizer[self.eos_token]  # 获取eos的id
        token_ids_0 = token_ids_0 + [gmask_id, self.sp_tokenizer[self.bos_token]]  # 添加gmask和bos到第一部分的尾部
        if token_ids_1 is not None:  # 如果存在第二部分
            token_ids_0 = token_ids_0 + token_ids_1 + [eos_id]  # 则将第二部分及eos添加到token_ids_0的尾部
        return token_ids_0  # 返回结果

    # 以下是与填充有关的方法
    def _pad(
            self,
            encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
            max_length: Optional[int] = None,
            padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
            pad_to_multiple_of: Optional[int] = None,
            return_attention_mask: Optional[bool] = None,
    ) -> dict:
        """
        Pad encoded inputs (on left/right and up to predefined length or max length in the batch)

        Args:
            encoded_inputs:
                Dictionary of tokenized inputs (`List[int]`) or batch of tokenized inputs (`List[List[int]]`).
            max_length: maximum length of the returned list and optionally padding length (see below).
                Will truncate by taking into account the special tokens.
            padding_strategy: PaddingStrategy to use for padding.

                - PaddingStrategy.LONGEST Pad to the longest sequence in the batch
                - PaddingStrategy.MAX_LENGTH: Pad to the max length (default)
                - PaddingStrategy.DO_NOT_PAD: Do not pad
                The tokenizer padding sides are defined in self.padding_side:

                    - 'left': pads on the left of the sequences
                    - 'right': pads on the right of the sequences
            pad_to_multiple_of: (optional) Integer if set will pad the sequence to a multiple of the provided value.
                This is especially useful to enable the use of Tensor Core on NVIDIA hardware with compute capability
                `>= 7.5` (Volta).
            return_attention_mask:
                (optional) Set to False to avoid returning attention mask (default: set to model specifics)
        """
        # Load from model defaults
        # 对编码后的输入进行填充
        bos_token_id = self.sp_tokenizer[self.bos_token]  # 获取bos的id
        mask_token_id = self.sp_tokenizer[self.mask_token]  # 获取mask的id
        gmask_token_id = self.sp_tokenizer[self.gmask_token]  # 获取gmask的id
        assert self.padding_side == "left"  # 断言填充在左边

        required_input = encoded_inputs[self.model_input_names[0]]  # 获取所需的输入
        seq_length = len(required_input)  # 获取序列长度

        if padding_strategy == PaddingStrategy.LONGEST:  # 如果填充策略是最长的
            max_length = len(required_input)  # 则最大长度为输入的长度

        # 如果最大长度不是pad_to_multiple_of的倍数，则进行相应的调整 
        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of  # 会得到大于 max_length 的一个整数，且可以被 pad_to_multiple_of 整除

        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length

        # Initialize attention mask if not present.
        if max_length is not None:
            if "attention_mask" not in encoded_inputs:
                if bos_token_id in required_input:
                    context_length = required_input.index(bos_token_id)
                else:
                    context_length = seq_length
                attention_mask = np.ones((1, seq_length, seq_length))
                attention_mask = np.tril(attention_mask)
                attention_mask[:, :, :context_length] = 1
                attention_mask = np.bool_(attention_mask < 0.5)
                encoded_inputs["attention_mask"] = attention_mask

            if "position_ids" not in encoded_inputs:
                if bos_token_id in required_input:
                    context_length = required_input.index(bos_token_id)
                else:
                    context_length = seq_length
                position_ids = np.arange(seq_length, dtype=np.int64)
                mask_token = mask_token_id if mask_token_id in required_input else gmask_token_id
                if mask_token in required_input:
                    mask_position = required_input.index(mask_token)
                    position_ids[context_length:] = mask_position
                block_position_ids = np.concatenate(
                    [np.zeros(context_length, dtype=np.int64),
                     np.arange(1, seq_length - context_length + 1, dtype=np.int64)])
                encoded_inputs["position_ids"] = np.stack([position_ids, block_position_ids], axis=0)

        if needs_to_be_padded:
            difference = max_length - len(required_input)

            if "attention_mask" in encoded_inputs:
                encoded_inputs["attention_mask"] = np.pad(encoded_inputs["attention_mask"],
                                                          pad_width=[(0, 0), (difference, 0), (difference, 0)],
                                                          mode='constant', constant_values=True)
            # - `pad_width`参数定义了每个维度上的填充大小。这里，我们不在第一维度上添加填充（通常是批次大小），而是在第二和第三维度的开始部分添加了`difference`数量的填充。
            # - `mode='constant'`表示填充模式为常量。
            # - `constant_values=True`表示使用True值进行填充，这在二进制遮罩中表示为1。
            if "token_type_ids" in encoded_inputs:
                encoded_inputs["token_type_ids"] = [self.pad_token_type_id] * difference + encoded_inputs[
                    "token_type_ids"
                ]
            if "special_tokens_mask" in encoded_inputs:
                encoded_inputs["special_tokens_mask"] = [1] * difference + encoded_inputs["special_tokens_mask"]
            if "position_ids" in encoded_inputs:
                encoded_inputs["position_ids"] = np.pad(encoded_inputs["position_ids"],
                                                        pad_width=[(0, 0), (difference, 0)])
            # - 同样，`pad_width`参数定义了每个维度上的填充大小。这里，我们不在第一维度上添加填充，而是在第二维度的开始部分添加了`difference`数量的填充。
            # - 这段代码没有明确指定填充的值，所以默认是0。
            encoded_inputs[self.model_input_names[0]] = [self.pad_token_id] * difference + required_input

        return encoded_inputs
