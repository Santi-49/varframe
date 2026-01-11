"""
VarFrame Exceptions
===================

Custom exceptions for VarFrame operations.
"""


class AmbiguityError(Exception):
    """
    Raised when multiple variable definitions match the same column name
    and no disambiguation is provided.
    
    This typically occurs when:
    - Multiple classes define variables with the same `name` attribute
    - Loading a CSV/Parquet file with conflicting variable definitions
    - The `ambiguity` parameter was not provided to resolve the conflict
    
    Example:
        >>> class MyVar(DerivedVariable):
        ...     name = "shared_name"
        ...     
        >>> class MyVarV2(DerivedVariable):
        ...     name = "shared_name"  # Collision!
        ...
        >>> # Loading will raise AmbiguityError
        >>> VarFrame.load_csv("data.csv")
        AmbiguityError: Ambiguous variable 'shared_name'. Multiple definitions found...
        
        >>> # Fix with explicit disambiguation
        >>> VarFrame.load_csv("data.csv", ambiguity={"shared_name": MyVarV2})
    """
    pass
