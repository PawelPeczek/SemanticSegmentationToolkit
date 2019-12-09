from typing import Optional


def prepare_block_operation_name(base_name: str,
                                 operation_name: str,
                                 postfix: Optional[str] = None) -> str:
    if postfix is None:
        postfix = ''
    else:
        postfix = f'/{postfix}'
    return f'{base_name}/{operation_name}{postfix}'
