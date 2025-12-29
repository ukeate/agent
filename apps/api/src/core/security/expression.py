import ast
from collections.abc import Mapping
from typing import Any, Callable

class UnsafeExpressionError(ValueError):
    ...

_ALLOWED_CALLS: dict[str, Callable[..., Any]] = {
    "abs": abs,
    "bool": bool,
    "float": float,
    "int": int,
    "len": len,
    "max": max,
    "min": min,
    "str": str,
}

def safe_eval(expression: str, names: Mapping[str, Any]) -> Any:
    try:
        node = ast.parse(expression, mode="eval").body
    except SyntaxError as e:
        raise UnsafeExpressionError(str(e)) from e
    return _eval(node, dict(names))

def safe_eval_bool(expression: str, names: Mapping[str, Any]) -> bool:
    value = safe_eval(expression, names)
    return bool(value)

def _eval(node: ast.AST, names: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value

    if isinstance(node, ast.Name):
        if node.id in names:
            value = names[node.id]
            if callable(value) and node.id not in _ALLOWED_CALLS:
                raise UnsafeExpressionError("禁止调用未授权函数")
            return value
        if node.id in _ALLOWED_CALLS:
            return _ALLOWED_CALLS[node.id]
        raise UnsafeExpressionError(f"未知变量: {node.id}")

    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            for v in node.values:
                if not _eval(v, names):
                    return False
            return True
        if isinstance(node.op, ast.Or):
            for v in node.values:
                if _eval(v, names):
                    return True
            return False
        raise UnsafeExpressionError("不支持的布尔运算")

    if isinstance(node, ast.UnaryOp):
        operand = _eval(node.operand, names)
        if isinstance(node.op, ast.Not):
            return not operand
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise UnsafeExpressionError("不支持的一元运算")

    if isinstance(node, ast.BinOp):
        left = _eval(node.left, names)
        right = _eval(node.right, names)
        op = node.op
        if isinstance(op, ast.Add):
            return left + right
        if isinstance(op, ast.Sub):
            return left - right
        if isinstance(op, ast.Mult):
            return left * right
        if isinstance(op, ast.Div):
            return left / right
        if isinstance(op, ast.Mod):
            return left % right
        if isinstance(op, ast.Pow):
            return left**right
        raise UnsafeExpressionError("不支持的二元运算")

    if isinstance(node, ast.Compare):
        left = _eval(node.left, names)
        for op, comparator in zip(node.ops, node.comparators, strict=True):
            right = _eval(comparator, names)
            ok: bool
            if isinstance(op, ast.Eq):
                ok = left == right
            elif isinstance(op, ast.NotEq):
                ok = left != right
            elif isinstance(op, ast.Lt):
                ok = left < right
            elif isinstance(op, ast.LtE):
                ok = left <= right
            elif isinstance(op, ast.Gt):
                ok = left > right
            elif isinstance(op, ast.GtE):
                ok = left >= right
            elif isinstance(op, ast.In):
                ok = left in right
            elif isinstance(op, ast.NotIn):
                ok = left not in right
            else:
                raise UnsafeExpressionError("不支持的比较运算")
            if not ok:
                return False
            left = right
        return True

    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name):
            if func.id not in _ALLOWED_CALLS:
                raise UnsafeExpressionError("禁止调用未授权函数")
            call = _ALLOWED_CALLS[func.id]
            args = [_eval(a, names) for a in node.args]
            kwargs = {kw.arg: _eval(kw.value, names) for kw in node.keywords}
            return call(*args, **kwargs)

        if isinstance(func, ast.Attribute) and func.attr == "get":
            base = _eval(func.value, names)
            if not isinstance(base, Mapping):
                raise UnsafeExpressionError("仅允许对字典调用 get")
            if len(node.args) not in {1, 2} or node.keywords:
                raise UnsafeExpressionError("不支持的 get 调用")
            key = _eval(node.args[0], names)
            default = _eval(node.args[1], names) if len(node.args) == 2 else None
            return base.get(key, default)

        raise UnsafeExpressionError("不支持的函数调用")

    if isinstance(node, ast.Subscript):
        base = _eval(node.value, names)
        slc = node.slice
        if isinstance(slc, ast.Slice):
            lower = _eval(slc.lower, names) if slc.lower else None
            upper = _eval(slc.upper, names) if slc.upper else None
            step = _eval(slc.step, names) if slc.step else None
            return base[slice(lower, upper, step)]
        idx = _eval(slc, names)
        return base[idx]

    if isinstance(node, ast.List):
        return [_eval(e, names) for e in node.elts]

    if isinstance(node, ast.Tuple):
        return tuple(_eval(e, names) for e in node.elts)

    if isinstance(node, ast.Dict):
        return {_eval(k, names): _eval(v, names) for k, v in zip(node.keys, node.values, strict=True)}

    raise UnsafeExpressionError("不支持的表达式")
