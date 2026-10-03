from langchain_core.globals import set_debug
from langchain_core.tracers import ConsoleCallbackHandler
from langchain_core.runnables import RunnableLambda

# Your chain
sequence = (
    RunnableLambda(lambda x: x + 1) |
    RunnableLambda(lambda x: x * 2)
)

# METHOD 1 - Debug everything globally
set_debug(True)
sequence.invoke(2)

# METHOD 2 - Debug only this one call
sequence.invoke(
    2,
    config={"callbacks": [ConsoleCallbackHandler()]}
)