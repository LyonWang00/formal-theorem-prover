# Error classes for Planner

class PlannerError(RuntimeError):
    """Planner 模块的基础异常。"""


class PlannerClientError(PlannerError):
    """模型调用或返回解析失败。"""


class BlueprintSchemaError(PlannerError):
    """Blueprint 不符合数据结构。"""


class BlueprintGraphError(PlannerError):
    """Blueprint 图结构不合法。"""


class LeanDeclarationError(PlannerError):
    """Lean declaration 检查失败。"""


class PlannerExhaustedError(PlannerError):
    """达到最大修复次数后仍失败。"""