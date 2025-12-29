"""代码文件解析器"""

import ast
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import chardet
from .base_parser import BaseParser, ParsedDocument, ParsedElement

logger = get_logger(__name__)

class CodeParser(BaseParser):
    """代码文件解析器
    
    支持多种编程语言的语法高亮和结构化解析
    """
    
    SUPPORTED_EXTENSIONS = [
        ".py", ".js", ".ts", ".jsx", ".tsx",  # Python, JavaScript, TypeScript
        ".java", ".cpp", ".c", ".h", ".hpp",  # Java, C/C++
        ".cs", ".go", ".rs", ".swift",        # C#, Go, Rust, Swift
        ".rb", ".php", ".scala", ".kt",       # Ruby, PHP, Scala, Kotlin
        ".sh", ".bash", ".yml", ".yaml",      # Shell, YAML
        ".json", ".xml", ".html", ".css",     # Data formats
        ".sql", ".r", ".m", ".lua"            # SQL, R, MATLAB, Lua
    ]
    
    # 语言特定的注释模式
    COMMENT_PATTERNS = {
        "python": (r'#.*$', r'"""[\s\S]*?"""', r"'''[\s\S]*?'''"),
        "javascript": (r'//.*$', r'/\*[\s\S]*?\*/'),
        "java": (r'//.*$', r'/\*[\s\S]*?\*/'),
        "c": (r'//.*$', r'/\*[\s\S]*?\*/'),
        "sql": (r'--.*$', r'/\*[\s\S]*?\*/'),
        "shell": (r'#.*$',),
        "html": (r'<!--[\s\S]*?-->',),
        "css": (r'/\*[\s\S]*?\*/',),
    }
    
    def __init__(
        self,
        extract_structure: bool = True,
        extract_docstrings: bool = True,
        syntax_highlight: bool = True
    ):
        """初始化代码解析器
        
        Args:
            extract_structure: 是否提取代码结构
            extract_docstrings: 是否提取文档字符串
            syntax_highlight: 是否进行语法高亮
        """
        super().__init__()
        self.extract_structure = extract_structure
        self.extract_docstrings = extract_docstrings
        self.syntax_highlight = syntax_highlight
    
    async def parse(self, file_path: Path) -> ParsedDocument:
        """解析代码文件
        
        Args:
            file_path: 代码文件路径
            
        Returns:
            解析后的文档
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        doc_id = self.generate_doc_id(file_path)
        metadata = self.extract_metadata(file_path)
        
        # 检测编码
        encoding = self._detect_encoding(file_path)
        
        # 读取文件内容
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
        except UnicodeDecodeError:
            # 降级到二进制模式
            with open(file_path, 'rb') as f:
                content = f.read().decode('utf-8', errors='ignore')
        
        # 识别编程语言
        language = self._identify_language(file_path)
        metadata["language"] = language
        metadata["encoding"] = encoding
        
        elements = []
        
        # 提取代码结构
        if self.extract_structure:
            if language == "python":
                structure_elements = await self._parse_python_structure(content, file_path)
                elements.extend(structure_elements)
            else:
                # 通用结构提取
                structure_elements = await self._parse_generic_structure(content, language)
                elements.extend(structure_elements)
        
        # 提取注释和文档
        comments = await self._extract_comments(content, language)
        if comments:
            elements.extend(comments)
        
        # 添加完整代码内容
        elements.append(ParsedElement(
            content=content,
            element_type="code",
            metadata={
                "language": language,
                "line_count": len(content.splitlines()),
                "char_count": len(content),
            }
        ))
        
        # 计算代码统计信息
        stats = self._calculate_code_stats(content, language)
        metadata.update(stats)
        
        return ParsedDocument(
            doc_id=doc_id,
            file_path=str(file_path),
            file_type="code",
            elements=elements,
            metadata=metadata
        )
    
    def _detect_encoding(self, file_path: Path) -> str:
        """检测文件编码
        
        Args:
            file_path: 文件路径
            
        Returns:
            编码类型
        """
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read()
                result = chardet.detect(raw_data)
                return result['encoding'] or 'utf-8'
        except:
            return 'utf-8'
    
    def _identify_language(self, file_path: Path) -> str:
        """识别编程语言
        
        Args:
            file_path: 文件路径
            
        Returns:
            语言标识符
        """
        ext = file_path.suffix.lower()
        
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.c': 'c',
            '.cpp': 'cpp',
            '.cs': 'csharp',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.php': 'php',
            '.sh': 'shell',
            '.bash': 'shell',
            '.yml': 'yaml',
            '.yaml': 'yaml',
            '.json': 'json',
            '.xml': 'xml',
            '.html': 'html',
            '.css': 'css',
            '.sql': 'sql',
            '.r': 'r',
            '.lua': 'lua',
        }
        
        return language_map.get(ext, 'text')
    
    async def _parse_python_structure(
        self, 
        content: str, 
        file_path: Path
    ) -> List[ParsedElement]:
        """解析Python代码结构
        
        Args:
            content: 代码内容
            file_path: 文件路径
            
        Returns:
            结构元素列表
        """
        elements = []
        
        try:
            tree = ast.parse(content)
            
            # 提取类定义
            classes = []
            functions = []
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_info = {
                        "name": node.name,
                        "line": node.lineno,
                        "methods": [],
                        "docstring": ast.get_docstring(node) if self.extract_docstrings else None
                    }
                    
                    # 提取方法
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            class_info["methods"].append({
                                "name": item.name,
                                "line": item.lineno,
                                "args": [arg.arg for arg in item.args.args],
                            })
                    
                    classes.append(class_info)
                
                elif isinstance(node, ast.FunctionDef) and not any(
                    isinstance(parent, ast.ClassDef) 
                    for parent in ast.walk(tree)
                ):
                    func_info = {
                        "name": node.name,
                        "line": node.lineno,
                        "args": [arg.arg for arg in node.args.args],
                        "docstring": ast.get_docstring(node) if self.extract_docstrings else None
                    }
                    functions.append(func_info)
                
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append(alias.name)
                    else:
                        module = node.module or ""
                        for alias in node.names:
                            imports.append(f"{module}.{alias.name}")
            
            # 创建结构摘要
            structure_lines = []
            
            if imports:
                structure_lines.append("## Imports")
                for imp in imports[:10]:  # 限制显示前10个
                    structure_lines.append(f"- {imp}")
                if len(imports) > 10:
                    structure_lines.append(f"... and {len(imports) - 10} more imports")
                structure_lines.append("")
            
            if classes:
                structure_lines.append("## Classes")
                for cls in classes:
                    structure_lines.append(f"- {cls['name']} (line {cls['line']})")
                    for method in cls["methods"][:5]:  # 限制显示前5个方法
                        structure_lines.append(f"  - {method['name']}({', '.join(method['args'])})")
                    if len(cls["methods"]) > 5:
                        structure_lines.append(f"  ... and {len(cls['methods']) - 5} more methods")
                structure_lines.append("")
            
            if functions:
                structure_lines.append("## Functions")
                for func in functions[:10]:  # 限制显示前10个函数
                    structure_lines.append(f"- {func['name']}({', '.join(func['args'])}) (line {func['line']})")
                if len(functions) > 10:
                    structure_lines.append(f"... and {len(functions) - 10} more functions")
            
            if structure_lines:
                elements.append(ParsedElement(
                    content="\n".join(structure_lines),
                    element_type="code_structure",
                    metadata={
                        "language": "python",
                        "classes_count": len(classes),
                        "functions_count": len(functions),
                        "imports_count": len(imports),
                        "classes": classes,
                        "functions": functions[:10],  # 存储前10个函数的详细信息
                    }
                ))
            
        except SyntaxError as e:
            logger.warning(f"Python syntax error in {file_path}: {e}")
            # 降级到基本解析
            elements.extend(await self._parse_generic_structure(content, "python"))
        
        return elements
    
    async def _parse_generic_structure(
        self, 
        content: str, 
        language: str
    ) -> List[ParsedElement]:
        """通用代码结构解析
        
        Args:
            content: 代码内容
            language: 语言类型
            
        Returns:
            结构元素列表
        """
        elements = []
        lines = content.splitlines()
        
        # 简单的函数/方法检测
        function_patterns = {
            "javascript": r'(function\s+(\w+)|const\s+(\w+)\s*=\s*(?:async\s+)?(?:function|\(.*?\)\s*=>))',
            "java": r'(?:public|private|protected)?\s*(?:static)?\s*\w+\s+(\w+)\s*\(',
            "c": r'\w+\s+(\w+)\s*\(',
            "go": r'func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\(',
            "rust": r'fn\s+(\w+)\s*(?:<.*?>)?\s*\(',
        }
        
        pattern = function_patterns.get(language)
        if pattern:
            functions = []
            for i, line in enumerate(lines, 1):
                match = re.search(pattern, line)
                if match:
                    func_name = match.group(1) or match.group(2) or match.group(3)
                    if func_name:
                        functions.append({
                            "name": func_name,
                            "line": i
                        })
            
            if functions:
                structure_lines = [f"## Functions/Methods found: {len(functions)}"]
                for func in functions[:15]:  # 限制显示前15个
                    structure_lines.append(f"- {func['name']} (line {func['line']})")
                if len(functions) > 15:
                    structure_lines.append(f"... and {len(functions) - 15} more")
                
                elements.append(ParsedElement(
                    content="\n".join(structure_lines),
                    element_type="code_structure",
                    metadata={
                        "language": language,
                        "functions_count": len(functions),
                    }
                ))
        
        return elements
    
    async def _extract_comments(
        self, 
        content: str, 
        language: str
    ) -> List[ParsedElement]:
        """提取代码注释
        
        Args:
            content: 代码内容
            language: 语言类型
            
        Returns:
            注释元素列表
        """
        elements = []
        
        # 获取语言特定的注释模式
        patterns = self.COMMENT_PATTERNS.get(
            language, 
            self.COMMENT_PATTERNS.get("c")  # 默认使用C风格注释
        )
        
        if not patterns:
            return elements
        
        all_comments = []
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.MULTILINE)
            all_comments.extend(matches)
        
        if all_comments:
            # 过滤和清理注释
            meaningful_comments = []
            for comment in all_comments[:50]:  # 限制处理前50个注释
                # 移除注释标记
                cleaned = re.sub(r'^[#/*\-\s]+|[*/\s]+$', '', comment).strip()
                # 过滤掉太短的注释
                if len(cleaned) > 10:
                    meaningful_comments.append(cleaned)
            
            if meaningful_comments:
                elements.append(ParsedElement(
                    content="\n".join(meaningful_comments[:20]),  # 限制显示前20个有意义的注释
                    element_type="comments",
                    metadata={
                        "language": language,
                        "total_comments": len(all_comments),
                        "meaningful_comments": len(meaningful_comments),
                    }
                ))
        
        return elements
    
    def _calculate_code_stats(self, content: str, language: str) -> Dict[str, Any]:
        """计算代码统计信息
        
        Args:
            content: 代码内容
            language: 语言类型
            
        Returns:
            统计信息字典
        """
        lines = content.splitlines()
        
        stats = {
            "total_lines": len(lines),
            "blank_lines": sum(1 for line in lines if not line.strip()),
            "code_lines": 0,
            "comment_lines": 0,
        }
        
        # 简单的注释行统计
        comment_prefixes = {
            "python": "#",
            "javascript": "//",
            "java": "//",
            "c": "//",
            "shell": "#",
            "sql": "--",
        }
        
        prefix = comment_prefixes.get(language)
        if prefix:
            for line in lines:
                stripped = line.strip()
                if stripped.startswith(prefix):
                    stats["comment_lines"] += 1
                elif stripped:
                    stats["code_lines"] += 1
        else:
            stats["code_lines"] = stats["total_lines"] - stats["blank_lines"]
        
        # 计算代码密度
        if stats["total_lines"] > 0:
            stats["code_density"] = round(
                stats["code_lines"] / stats["total_lines"] * 100, 2
            )
        else:
            stats["code_density"] = 0
        
        return stats
from src.core.logging import get_logger
