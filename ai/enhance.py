import os
import json
import sys
import re
import hashlib
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional
from queue import Queue
from threading import Lock
from pathlib import Path
# INSERT_YOUR_CODE
import requests

import dotenv
import argparse
from tqdm import tqdm

# 尝试导入 json5，如果不可用则回退到标准 json
try:
    import json5
    HAS_JSON5 = True
except ImportError:
    HAS_JSON5 = False

import langchain_core.exceptions
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from structure import Structure

if os.path.exists('.env'):
    dotenv.load_dotenv()
template = open("template.txt", "r").read()
system = open("system.txt", "r").read()

# 默认 AI 字段值
DEFAULT_AI_FIELDS = {
    "tldr": "摘要生成失败",
    "motivation": "动机分析不可用",
    "method": "方法提取失败",
    "result": "结果分析不可用",
    "conclusion": "结论提取失败"
}


class AICache:
    """
    AI 处理结果缓存管理器
    
    缓存策略:
    - 使用 JSONL 格式存储，每行一个论文的缓存
    - 缓存 key 由 arxiv_id + template_hash 组成，确保模板更新后重新处理
    - 支持增量处理：只处理未缓存或缓存失效的论文
    """
    
    def __init__(self, cache_dir: str = None, template_file: str = "template.txt", system_file: str = "system.txt"):
        """
        初始化缓存管理器
        
        Args:
            cache_dir: 缓存目录路径，默认为 ../data/ai_cache
            template_file: 模板文件路径，用于计算模板哈希
            system_file: 系统提示文件路径，用于计算模板哈希
        """
        if cache_dir is None:
            # 默认缓存目录在 data/ai_cache
            cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "ai_cache")
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 计算模板哈希，用于缓存失效判断
        self.template_hash = self._compute_template_hash(template_file, system_file)
        
        # 内存缓存
        self._cache: Dict[str, Dict] = {}
        self._cache_loaded = False
        self._lock = Lock()
        
        # 失败记录
        self._failed_items: List[Dict] = []
    
    def _compute_template_hash(self, template_file: str, system_file: str) -> str:
        """计算模板文件的哈希值"""
        hasher = hashlib.md5()
        
        for filepath in [template_file, system_file]:
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    hasher.update(f.read())
        
        return hasher.hexdigest()[:8]
    
    def _get_cache_file(self, date: str = None) -> Path:
        """获取缓存文件路径"""
        if date is None:
            date = datetime.utcnow().strftime("%Y-%m-%d")
        return self.cache_dir / f"cache_{date}.jsonl"
    
    def _get_failed_file(self, date: str = None) -> Path:
        """获取失败记录文件路径"""
        if date is None:
            date = datetime.utcnow().strftime("%Y-%m-%d")
        return self.cache_dir / f"failed_{date}.jsonl"
    
    def load_cache(self, date: str = None) -> None:
        """加载缓存文件到内存"""
        cache_file = self._get_cache_file(date)
        
        with self._lock:
            self._cache.clear()
            
            if cache_file.exists():
                with open(cache_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            item = json.loads(line.strip())
                            arxiv_id = item.get('id')
                            tmpl_hash = item.get('_template_hash', '')
                            
                            # 只有模板哈希匹配的缓存才有效
                            if arxiv_id and tmpl_hash == self.template_hash:
                                self._cache[arxiv_id] = item
                        except json.JSONDecodeError:
                            continue
                
                print(f"Loaded {len(self._cache)} valid cached items from {cache_file}", file=sys.stderr)
            
            self._cache_loaded = True
    
    def get(self, arxiv_id: str) -> Optional[Dict]:
        """获取缓存的处理结果"""
        if not self._cache_loaded:
            self.load_cache()
        
        with self._lock:
            cached = self._cache.get(arxiv_id)
            if cached:
                # 检查是否有有效的 AI 字段（不是默认值）
                ai_data = cached.get('AI', {})
                if ai_data and ai_data.get('tldr') != DEFAULT_AI_FIELDS['tldr']:
                    return cached
            return None
    
    def set(self, item: Dict) -> None:
        """保存处理结果到缓存"""
        arxiv_id = item.get('id')
        if not arxiv_id:
            return
        
        # 添加模板哈希和时间戳
        item['_template_hash'] = self.template_hash
        item['_cached_at'] = datetime.utcnow().isoformat()
        
        with self._lock:
            self._cache[arxiv_id] = item
    
    def save_cache(self, date: str = None) -> None:
        """保存缓存到文件"""
        cache_file = self._get_cache_file(date)
        
        with self._lock:
            with open(cache_file, 'w', encoding='utf-8') as f:
                for item in self._cache.values():
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            print(f"Saved {len(self._cache)} items to cache: {cache_file}", file=sys.stderr)
    
    def record_failure(self, item: Dict, reason: str) -> None:
        """记录处理失败的论文"""
        failure_record = {
            'id': item.get('id'),
            'title': item.get('title', ''),
            'reason': reason,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        with self._lock:
            self._failed_items.append(failure_record)
    
    def save_failures(self, date: str = None) -> None:
        """保存失败记录到文件"""
        if not self._failed_items:
            return
        
        failed_file = self._get_failed_file(date)
        
        with self._lock:
            with open(failed_file, 'w', encoding='utf-8') as f:
                for item in self._failed_items:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            print(f"Saved {len(self._failed_items)} failure records to: {failed_file}", file=sys.stderr)
    
    def get_failed_ids(self, date: str = None) -> List[str]:
        """获取失败的论文 ID 列表"""
        failed_file = self._get_failed_file(date)
        failed_ids = []
        
        if failed_file.exists():
            with open(failed_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        if item.get('id'):
                            failed_ids.append(item['id'])
                    except json.JSONDecodeError:
                        continue
        
        return failed_ids
    
    def is_cached(self, arxiv_id: str) -> bool:
        """检查论文是否已缓存"""
        return self.get(arxiv_id) is not None
    
    def get_stats(self) -> Dict:
        """获取缓存统计信息"""
        return {
            'total_cached': len(self._cache),
            'total_failed': len(self._failed_items),
            'template_hash': self.template_hash,
            'cache_dir': str(self.cache_dir)
        }


def fix_json_string(json_str: str) -> str:
    """
    修复常见的 JSON 格式问题
    
    处理以下情况:
    - LaTeX 中的反斜杠
    - 未转义的换行符
    - 控制字符
    - 不完整的 JSON 结构
    """
    if not json_str:
        return "{}"
    
    # 1. 处理 LaTeX 中的反斜杠 (只转义非标准转义序列)
    # 标准 JSON 转义: \", \\, \/, \b, \f, \n, \r, \t, \uXXXX
    json_str = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', json_str)
    
    # 2. 处理未转义的换行符和特殊字符
    # 需要在字符串值内部处理，避免破坏 JSON 结构
    def escape_string_content(match):
        content = match.group(1)
        content = content.replace('\n', '\\n')
        content = content.replace('\r', '\\r')
        content = content.replace('\t', '\\t')
        return f'"{content}"'
    
    # 匹配 JSON 字符串值 (简单版本)
    json_str = re.sub(r'"((?:[^"\\]|\\.)*?)"', escape_string_content, json_str, flags=re.DOTALL)
    
    # 3. 移除控制字符
    json_str = re.sub(r'[\x00-\x1f](?<![\n\r\t])', '', json_str)
    
    # 4. 尝试修复不完整的 JSON
    json_str = json_str.strip()
    
    # 修复缺少结束引号
    if json_str.count('"') % 2 != 0:
        json_str += '"'
    
    # 修复缺少结束括号
    open_braces = json_str.count('{') - json_str.count('}')
    if open_braces > 0:
        json_str += '}' * open_braces
    
    return json_str


def extract_fields_with_regex(text: str) -> Dict[str, str]:
    """
    使用正则表达式从文本中提取各字段
    作为 JSON 解析失败时的回退方案
    """
    result = {}
    
    # 定义各字段的匹配模式
    field_patterns = {
        'tldr': [
            r'"tldr"\s*:\s*"((?:[^"\\]|\\.)*)"',
            r'tldr[:\s]+([^\n]+)',
            r'TL;?DR[:\s]+([^\n]+)',
        ],
        'motivation': [
            r'"motivation"\s*:\s*"((?:[^"\\]|\\.)*)"',
            r'motivation[:\s]+([^\n]+)',
        ],
        'method': [
            r'"method"\s*:\s*"((?:[^"\\]|\\.)*)"',
            r'method[:\s]+([^\n]+)',
        ],
        'result': [
            r'"result"\s*:\s*"((?:[^"\\]|\\.)*)"',
            r'result[:\s]+([^\n]+)',
        ],
        'conclusion': [
            r'"conclusion"\s*:\s*"((?:[^"\\]|\\.)*)"',
            r'conclusion[:\s]+([^\n]+)',
        ],
    }
    
    for field, patterns in field_patterns.items():
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                value = match.group(1).strip()
                # 清理提取的值
                value = value.strip('"\'')
                value = re.sub(r'\s+', ' ', value)
                if value and len(value) > 5:  # 确保有意义的内容
                    result[field] = value
                    break
    
    return result


def robust_parse_json(json_str: str, default_fields: Dict[str, str]) -> Dict[str, str]:
    """
    使用多重策略解析 JSON 字符串
    
    策略顺序:
    1. 标准 JSON 解析
    2. 修复后的 JSON 解析
    3. json5 宽容解析 (如果可用)
    4. 正则表达式提取
    5. 返回默认值
    """
    if not json_str:
        return default_fields.copy()
    
    # 策略 1: 尝试标准 JSON 解析
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    
    # 策略 2: 修复后重新解析
    fixed_json = fix_json_string(json_str)
    try:
        return json.loads(fixed_json)
    except json.JSONDecodeError:
        pass
    
    # 策略 3: 使用 json5 进行宽容解析 (支持尾逗号、注释等)
    if HAS_JSON5:
        try:
            return json5.loads(json_str)
        except:
            pass
        try:
            return json5.loads(fixed_json)
        except:
            pass
    
    # 策略 4: 使用正则表达式提取字段
    extracted = extract_fields_with_regex(json_str)
    if extracted:
        print(f"Used regex extraction, found fields: {list(extracted.keys())}", file=sys.stderr)
        return {**default_fields, **extracted}
    
    # 策略 5: 返回默认值
    return default_fields.copy()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="jsonline data file")
    parser.add_argument("--max_workers", type=int, default=1, help="Maximum number of parallel workers")
    parser.add_argument("--use-cache", action="store_true", default=True, help="Use AI result cache (default: True)")
    parser.add_argument("--no-cache", action="store_true", help="Disable AI result cache")
    parser.add_argument("--retry-failed", action="store_true", help="Only retry previously failed items")
    parser.add_argument("--cache-dir", type=str, default=None, help="Custom cache directory path")
    return parser.parse_args()

def process_single_item(chain, item: Dict, language: str) -> Dict:
    def is_sensitive(content: str) -> bool:
        """
        调用 spam.dw-dengwei.workers.dev 接口检测内容是否包含敏感词。
        返回 True 表示触发敏感词，False 表示未触发。
        """
        try:
            resp = requests.post(
                "https://spam.dw-dengwei.workers.dev",
                json={"text": content},
                timeout=5
            )
            if resp.status_code == 200:
                result = resp.json()
                # 约定接口返回 {"sensitive": true/false, ...}
                return result.get("sensitive", True)
            else:
                # 如果接口异常，默认不触发敏感词
                print(f"Sensitive check failed with status {resp.status_code}", file=sys.stderr)
                return True
        except Exception as e:
            print(f"Sensitive check error: {e}", file=sys.stderr)
            return True

    def check_github_code(content: str) -> Dict:
        """提取并验证 GitHub 链接"""
        code_info = {}

        # 1. 优先匹配 github.com/owner/repo 格式
        github_pattern = r"https?://github\.com/([a-zA-Z0-9-_]+)/([a-zA-Z0-9-_\.]+)"
        match = re.search(github_pattern, content)
        
        if match:
            owner, repo = match.groups()
            # 清理 repo 名称，去掉可能的 .git 后缀或末尾的标点
            repo = repo.rstrip(".git").rstrip(".,)")
            
            full_url = f"https://github.com/{owner}/{repo}"
            code_info["code_url"] = full_url
            
            # 尝试调用 GitHub API 获取信息
            github_token = os.environ.get("TOKEN_GITHUB")
            headers = {"Accept": "application/vnd.github.v3+json"}
            if github_token:
                headers["Authorization"] = f"token {github_token}"
            
            try:
                api_url = f"https://api.github.com/repos/{owner}/{repo}"
                resp = requests.get(api_url, headers=headers, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    code_info["code_stars"] = data.get("stargazers_count", 0)
                    code_info["code_last_update"] = data.get("pushed_at", "")[:10]
            except Exception:
                # API 调用失败不影响主流程
                pass
            return code_info

        # 2. 如果没有 github.com，尝试匹配 github.io
        github_io_pattern = r"https?://[a-zA-Z0-9-_]+\.github\.io(?:/[a-zA-Z0-9-_\.]+)*"
        match_io = re.search(github_io_pattern, content)
        
        if match_io:
            url = match_io.group(0)
            # 清理末尾标点
            url = url.rstrip(".,)")
            code_info["code_url"] = url
            # github.io 不进行 star 和 update 判断
                
        return code_info

    # 检查 summary 字段
    if is_sensitive(item.get("summary", "")):
        return None

    # 检测代码可用性
    code_info = check_github_code(item.get("summary", ""))
    if code_info:
        item.update(code_info)

    """处理单个数据项"""
    try:
        response: Structure = chain.invoke({
            "language": language,
            "content": item['summary']
        })
        item['AI'] = response.model_dump()
    except langchain_core.exceptions.OutputParserException as e:
        # 尝试从错误信息中提取并修复 JSON
        error_msg = str(e)
        partial_data = {}
        
        # 尝试多种方式提取 JSON
        json_candidates = []
        
        # 方式1: 从 "Function Structure arguments:" 后提取
        if "Function Structure arguments:" in error_msg:
            json_str = error_msg.split("Function Structure arguments:", 1)[1]
            json_str = json_str.split('are not valid JSON')[0].strip()
            json_candidates.append(json_str)
        
        # 方式2: 从 "Invalid json output:" 后提取
        if "Invalid json output:" in error_msg:
            json_str = error_msg.split("Invalid json output:", 1)[1].strip()
            json_candidates.append(json_str)
        
        # 方式3: 尝试直接从错误信息中找 JSON 对象
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', error_msg, re.DOTALL)
        if json_match:
            json_candidates.append(json_match.group(0))
        
        # 尝试解析每个候选 JSON
        for candidate in json_candidates:
            if candidate:
                parsed = robust_parse_json(candidate, DEFAULT_AI_FIELDS)
                if parsed and parsed != DEFAULT_AI_FIELDS:
                    partial_data = parsed
                    break
        
        # 如果所有方式都失败，尝试从整个错误信息中提取字段
        if not partial_data or partial_data == DEFAULT_AI_FIELDS:
            partial_data = extract_fields_with_regex(error_msg)
        
        item['AI'] = {**DEFAULT_AI_FIELDS, **partial_data}
        if partial_data and partial_data != DEFAULT_AI_FIELDS:
            print(f"Recovered partial AI data for {item.get('id', 'unknown')}: {list(partial_data.keys())}", file=sys.stderr)
        else:
            print(f"Using default AI data for {item.get('id', 'unknown')} due to parse error", file=sys.stderr)
            
    except Exception as e:
        # Catch any other exceptions and provide default values
        print(f"Unexpected error for {item.get('id', 'unknown')}: {type(e).__name__}: {e}", file=sys.stderr)
        item['AI'] = DEFAULT_AI_FIELDS.copy()
    
    # Final validation to ensure all required fields exist
    for field, default_value in DEFAULT_AI_FIELDS.items():
        if field not in item['AI'] or not item['AI'][field]:
            item['AI'][field] = default_value

    # 检查 AI 生成的所有字段
    for v in item.get("AI", {}).values():
        if is_sensitive(str(v)):
            return None
    return item

def process_all_items(data: List[Dict], model_name: str, language: str, max_workers: int, cache: Optional[AICache] = None) -> List[Dict]:
    """并行处理所有数据项，支持缓存"""
    llm = ChatOpenAI(model=model_name).with_structured_output(Structure, method="function_calling")
    print('Connect to:', model_name, file=sys.stderr)
    
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system),
        HumanMessagePromptTemplate.from_template(template=template)
    ])

    chain = prompt_template | llm
    
    # 分离已缓存和需要处理的数据
    items_to_process = []
    processed_data = [None] * len(data)
    cached_count = 0
    
    for idx, item in enumerate(data):
        arxiv_id = item.get('id')
        
        # 检查缓存
        if cache:
            cached_item = cache.get(arxiv_id)
            if cached_item:
                # 使用缓存结果，但更新原始数据中可能变化的字段
                cached_item.update({
                    'pdf': item.get('pdf'),
                    'abs': item.get('abs'),
                    'authors': item.get('authors'),
                    'categories': item.get('categories'),
                    'comment': item.get('comment'),
                })
                processed_data[idx] = cached_item
                cached_count += 1
                continue
        
        items_to_process.append((idx, item))
    
    if cached_count > 0:
        print(f"Using {cached_count} cached results, processing {len(items_to_process)} new items", file=sys.stderr)
    
    # 如果所有数据都已缓存，直接返回
    if not items_to_process:
        print("All items were cached, no LLM calls needed!", file=sys.stderr)
        return processed_data
    
    # 使用线程池并行处理未缓存的数据
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_idx = {
            executor.submit(process_single_item, chain, item, language): idx
            for idx, item in items_to_process
        }
        
        # 使用tqdm显示进度
        for future in tqdm(
            as_completed(future_to_idx),
            total=len(items_to_process),
            desc="Processing items"
        ):
            idx = future_to_idx[future]
            original_item = data[idx]
            
            try:
                result = future.result()
                processed_data[idx] = result
                
                # 保存成功结果到缓存
                if cache and result:
                    cache.set(result)
                elif cache and result is None:
                    # 记录被过滤（敏感词）的情况
                    cache.record_failure(original_item, "Filtered by sensitive word check")
                    
            except Exception as e:
                print(f"Item at index {idx} generated an exception: {e}", file=sys.stderr)
                # Add default AI fields to ensure consistency
                processed_data[idx] = original_item
                processed_data[idx]['AI'] = DEFAULT_AI_FIELDS.copy()
                
                # 记录失败
                if cache:
                    cache.record_failure(original_item, f"Exception: {type(e).__name__}: {str(e)}")
    
    return processed_data

def main():
    args = parse_args()
    model_name = os.environ.get("MODEL_NAME", 'deepseek-chat')
    language = os.environ.get("LANGUAGE", 'Chinese')

    # 检查并删除目标文件
    target_file = args.data.replace('.jsonl', f'_AI_enhanced_{language}.jsonl')
    if os.path.exists(target_file):
        os.remove(target_file)
        print(f'Removed existing file: {target_file}', file=sys.stderr)

    # 初始化缓存（除非明确禁用）
    cache = None
    use_cache = args.use_cache and not args.no_cache
    
    if use_cache:
        cache = AICache(cache_dir=args.cache_dir)
        
        # 从数据文件名提取日期
        data_filename = os.path.basename(args.data)
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', data_filename)
        date_str = date_match.group(1) if date_match else None
        
        cache.load_cache(date_str)
        print(f"Cache initialized: {cache.get_stats()}", file=sys.stderr)

    # 读取数据
    data = []
    with open(args.data, "r") as f:
        for line in f:
            data.append(json.loads(line))

    # 去重
    seen_ids = set()
    unique_data = []
    for item in data:
        if item['id'] not in seen_ids:
            seen_ids.add(item['id'])
            unique_data.append(item)

    data = unique_data
    print('Open:', args.data, file=sys.stderr)
    
    # 如果是重试失败模式，只处理之前失败的论文
    if args.retry_failed and cache:
        data_filename = os.path.basename(args.data)
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', data_filename)
        date_str = date_match.group(1) if date_match else None
        
        failed_ids = set(cache.get_failed_ids(date_str))
        if failed_ids:
            data = [item for item in data if item['id'] in failed_ids]
            print(f"Retry mode: processing {len(data)} previously failed items", file=sys.stderr)
        else:
            print("No failed items to retry", file=sys.stderr)
    
    # 并行处理所有数据
    processed_data = process_all_items(
        data,
        model_name,
        language,
        args.max_workers,
        cache=cache
    )
    
    # 保存缓存
    if cache:
        data_filename = os.path.basename(args.data)
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', data_filename)
        date_str = date_match.group(1) if date_match else None
        
        cache.save_cache(date_str)
        cache.save_failures(date_str)
        
        stats = cache.get_stats()
        print(f"Final cache stats: {stats}", file=sys.stderr)
    
    # 保存结果
    with open(target_file, "w") as f:
        for item in processed_data:
            if item is not None:
                # 移除内部缓存字段
                item_to_save = {k: v for k, v in item.items() if not k.startswith('_')}
                f.write(json.dumps(item_to_save) + "\n")

if __name__ == "__main__":
    main()
