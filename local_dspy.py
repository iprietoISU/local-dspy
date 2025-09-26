from xml.parsers.expat import model
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import dspy

import logging
import os
import re
import threading
import warnings
from typing import Any, Literal, cast

from unsloth import FastLanguageModel, FastModel, FastLlamaModel

import dspy
from dspy.clients.cache import request_cache
from dspy.clients.openai import OpenAIProvider
from dspy.clients.provider import Provider, ReinforceJob, TrainingJob
from dspy.clients.utils_finetune import TrainDataFormat
from dspy.dsp.utils.settings import settings
from dspy.utils.callback import BaseCallback

from dspy.clients.base_lm import BaseLM

from munch import munchify

from langchain_community.llms.llamacpp import LlamaCpp

logger = logging.getLogger(__name__)
FORMAT_MARKER = "!MARKER_FOR_RESPONSE_FORMATTING!"  # Marker to help models format responses correctly.
FINELOAD = False

class HyperLocalLocalProvider(Provider):

    @staticmethod
    def perform_completion(request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None = None):
        raise NotImplementedError("Logic not implemented/found for perform_completion.")


class UnslothLocalProvider(HyperLocalLocalProvider):
    model_instance: FastLanguageModel | None = None
    tokenizer_instance = None

    @staticmethod
    def launch(lm: BaseLM, launch_kwargs: dict[str, Any] | None = None):
        print("Unsloth compatible model detected. Using Triton Kernels.")

        UnslothLocalProvider.model_instance, UnslothLocalProvider.tokenizer_instance = (
            FastLanguageModel.from_pretrained(
                model_name=lm.model,
                max_seq_length=lm.kwargs.get("max_tokens", 1024),
                dtype=lm.kwargs.get("dtype", None),
                load_in_4bit=lm.kwargs.get("load_in_4bit", True),
                cache_dir="aisystem/ai_cache",
            )
        )

        FastLanguageModel.for_inference(
            UnslothLocalProvider.model_instance
        )  # Enable native 2x faster inference

    @staticmethod
    def kill(lm: BaseLM, launch_kwargs: dict[str, Any] | None = None):
        UnslothLocalProvider.model_instance = None
        UnslothLocalProvider.tokenizer_instance = None
        print("Unsloth model instance set to None.")

    @staticmethod
    def perform_completion(request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None = None):
        if (
            UnslothLocalProvider.model_instance is None
            or UnslothLocalProvider.tokenizer_instance is None
        ):
            raise ValueError(
                "Unsloth model is not launched. Please call launch() before performing completion."
            )

        #print("A", request["messages"])
        inputs = UnslothLocalProvider.tokenizer_instance.apply_chat_template(
            request["messages"], tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")
        #print("B")
        #print("\n\n\n Trying to generate with:", inputs)
        outputs = UnslothLocalProvider.model_instance.generate(
            input_ids=inputs,
            max_new_tokens=request.get(
                "max_tokens", 1024
            ),  # You can adjust max_new_tokens as needed
        )

        #print("Generated outputs:", outputs)

        response = UnslothLocalProvider.tokenizer_instance.decode(
            outputs[0], skip_special_tokens=True
        )

        #print("Successfully decoded response:", response)

        # return {
        #     "choices": [{"message": {"role": "assistant", "content": response}, "finish_reason": "stop"}],
        #     "usage": {"prompt_tokens": inputs['input_ids'].shape[1], "completion_tokens": len(response.split()), "total_tokens": inputs['input_ids'].shape[1] + len(response.split())}
        # }

        return {
            "id": "null",
            "object": "chat.completion",
            "created": 1741569952,
            "model": request["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response.split(FORMAT_MARKER)[-1],
                        "refusal": None,
                        "annotations": [],
                    },
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": {},
            "service_tier": "default",
        }
    

class LPCPPLocalProvider(HyperLocalLocalProvider):
    model_instance: LlamaCpp | Llama | None = None

    @staticmethod
    def launch(lm: BaseLM, launch_kwargs: dict[str, Any] | None = None):

        model_path = lm.model
        if "::" in lm.model:
            base_model, gguf = lm.model.split("::", 1)
            model_path = hf_hub_download(
                repo_id=base_model,
                filename=gguf,
                cache_dir="aisystem/ai_cache",
            )

        LPCPPLocalProvider.model_instance = llm = LlamaCpp(
            model_path=model_path,
            n_ctx=lm.kwargs.get("max_tokens", 8192)*2, # Context length
            n_batch=512,            # Batch size
            n_threads=8,            # CPU threads
            n_gpu_layers=40,        # Set to 0 for CPU-only
            f16_kv=True,            # Use fp16 for key/values
            verbose=False,
            temperature=1.0,
            max_tokens=lm.kwargs.get("max_tokens", 8192),
            top_k=64,
            top_p=0.95,
            repeat_penalty=1.1,

            #stop=["<start_of_turn>"],
        ) if FINELOAD else Llama.from_pretrained(
            repo_id="unsloth/gpt-oss-20b-GGUF",
            filename="gpt-oss-20b-UD-Q4_K_XL.gguf",
            cache_dir="aisystem/ai_cache",
            n_ctx=lm.kwargs.get("max_tokens", 8192)*2, # Context length
            #n_batch=512,            # Batch size
            #n_threads=8,            # CPU threads
            n_gpu_layers=15,        # Set to 0 for CPU-only
            #f16_kv=True,            # Use fp16 for key/values
            verbose=True,
            temperature=lm.kwargs.get("temperature", 0.7),
            max_tokens=lm.kwargs.get("max_tokens", 8192),
            top_k=lm.kwargs.get("top_k", 64),
            top_p=lm.kwargs.get("top_p", 0.95),
            #repeat_penalty=1.1,
        )


    @staticmethod
    def kill(lm: BaseLM, launch_kwargs: dict[str, Any] | None = None):
        LPCPPLocalProvider.model_instance = None
        print("LPCPP model instance set to None.")

    @staticmethod
    def perform_completion(request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None = None):
        if (
            LPCPPLocalProvider.model_instance is None
        ):
            raise ValueError(
                "LPCPP model is not launched. Please call launch() before performing completion."
            )
        

        print("Beeeee...")

        if FINELOAD:
            output = LPCPPLocalProvider.model_instance.invoke(request["messages"])

            return {
                "id": "null",
                "object": "chat.completion",
                "created": 1741569952,
                "model": request["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": output,
                            "refusal": None,
                            "annotations": [],
                        },
                        "logprobs": None,
                        "finish_reason": "stop",
                    }
                ],
                "usage": {},
                "service_tier": "default",
            }
        else:
            output = LPCPPLocalProvider.model_instance.create_chat_completion(
                messages=request["messages"], stop=["[[ ## completed ## ]]"]
            )
            output = munchify(output)

            #TODO: Make this dependant on kwargs
            output.choices[0].message.content = output.choices[0].message.content.split("<|channel|>final<|message|>")[-1]
            return output


class HyperLocalLM(BaseLM):
    """
    Enhanced Language Model class that provides the ability for dspy to execute on the edge.
    Utilizes Unsloth and Llama.cpp for local model inference, depending on the model format.
    """

    def __init__(
        self,
        model: str,
        model_type: Literal["chat"] = "chat",
        temperature: float = 0.0,
        max_tokens: int = 16000,
        cache: bool = False,
        callbacks: list[BaseCallback] | None = None,
        num_retries: int = 3,
        provider: HyperLocalLocalProvider | None = None,
        finetuning_model: str | None = None,
        launch_kwargs: dict[str, Any] | None = None,
        train_kwargs: dict[str, Any] | None = None,
        use_developer_role: bool = False,
        **kwargs,
    ):
        """
        Create a new language model instance for use with DSPy modules and programs. Adapted from dspy.LM.

        Args:
            model: The model to use. This should be a string of the form ``"llm_provider/llm_name"``
                   supported by LiteLLM. For example, ``"openai/gpt-4o"``.
            model_type: The type of the model, either ``"chat"`` or ``"text"``.
            temperature: The sampling temperature to use when generating responses.
            max_tokens: The maximum number of tokens to generate per response.
            cache: Whether to cache the model responses for reuse to improve performance
                   and reduce costs.
            callbacks: A list of callback functions to run before and after each request.
            num_retries: The number of times to retry a request if it fails transiently due to
                         network error, rate limiting, etc. Requests are retried with exponential
                         backoff.
            provider: The provider to use. If not specified, the provider will be inferred from the model.
            finetuning_model: The model to finetune. In some providers, the models available for finetuning is different
                from the models available for inference.
            rollout_id: Optional integer used to differentiate cache entries for otherwise
                identical requests. Different values bypass DSPy's caches while still caching
                future calls with the same inputs and rollout ID. Note that `rollout_id`
                only affects generation when `temperature` is non-zero. This argument is
                stripped before sending requests to the provider.
        """
        # Remember to update LM.copy() if you modify the constructor!
        self.model = model
        self.model_type = model_type
        self.cache = cache
        self.provider: HyperLocalLocalProvider = provider or self.infer_provider()
        self.callbacks = callbacks or []
        self.history = []
        self.num_retries = num_retries
        self.finetuning_model = finetuning_model
        self.launch_kwargs = launch_kwargs or {}
        self.train_kwargs = train_kwargs or {}
        self.use_developer_role = use_developer_role
        self._warned_zero_temp_rollout = False

        # Handle model-specific configuration for different model families
        model_family = model.split("/")[-1].lower() if "/" in model else model.lower()

        # Recognize OpenAI reasoning models (o1, o3, o4, gpt-5 family)
        model_pattern = re.match(r"^(?:o[1345]|gpt-5)(?:-(?:mini|nano))?", model_family)

        if model_pattern:
            if max_tokens < 16000 or temperature != 1.0:
                raise ValueError(
                    "OpenAI's reasoning models require passing temperature=1.0 and max_tokens >= 16000 to "
                    "`dspy.LM(...)`, e.g., dspy.LM('openai/gpt-5', temperature=1.0, max_tokens=16000)"
                )
            self.kwargs = dict(
                temperature=temperature, max_completion_tokens=max_tokens, **kwargs
            )
            if self.kwargs.get("rollout_id") is None:
                self.kwargs.pop("rollout_id", None)
        else:
            self.kwargs = dict(temperature=temperature, max_tokens=max_tokens, **kwargs)
            if self.kwargs.get("rollout_id") is None:
                self.kwargs.pop("rollout_id", None)

        self._warn_zero_temp_rollout(
            self.kwargs.get("temperature"), self.kwargs.get("rollout_id")
        )

    def _warn_zero_temp_rollout(self, temperature: float | None, rollout_id):
        if (
            not self._warned_zero_temp_rollout
            and rollout_id is not None
            and (temperature is None or temperature == 0)
        ):
            warnings.warn(
                "rollout_id has no effect when temperature=0; set temperature>0 to bypass the cache.",
                stacklevel=3,
            )
            self._warned_zero_temp_rollout = True

    def _get_cached_completion_fn(self, completion_fn, cache):
        ignored_args_for_cache_key = ["api_key", "api_base", "base_url"]
        if cache:
            completion_fn = request_cache(
                cache_arg_name="request",
                ignored_args_for_cache_key=ignored_args_for_cache_key,
            )(completion_fn)

        litellm_cache_args = {"no-cache": True, "no-store": True}

        return completion_fn, litellm_cache_args

    def forward(self, prompt=None, messages=None, **kwargs):
        # Build the request.
        kwargs = dict(kwargs)
        cache = kwargs.pop("cache", self.cache)

        messages = messages or [{"role": "user", "content": prompt}]
        if self.use_developer_role and self.model_type == "responses":
            messages = [
                {**m, "role": "developer"} if m.get("role") == "system" else m
                for m in messages
            ]
        kwargs = {**self.kwargs, **kwargs}
        self._warn_zero_temp_rollout(
            kwargs.get("temperature"), kwargs.get("rollout_id")
        )
        if kwargs.get("rollout_id") is None:
            kwargs.pop("rollout_id", None)

        completion = self.provider.perform_completion

        completion, litellm_cache_args = self._get_cached_completion_fn(
            completion, cache
        )

        messages[-1]["content"] += FORMAT_MARKER
        # print("MESSAGE SENT IN",messages)
        results = completion(
            request=dict(model=self.model, messages=messages, **kwargs),
            num_retries=self.num_retries,
            cache=litellm_cache_args,
        )

        results = munchify(results)

        #print(results.choices)
        #print(results.choices[0].message.content)
        #exit()

        self._check_truncation(results)

        if (
            not getattr(results, "cache_hit", False)
            and dspy.settings.usage_tracker
            and hasattr(results, "usage")
        ):
            settings.usage_tracker.add_usage(self.model, dict(results.usage))
        
        return results

    def launch(self, launch_kwargs: dict[str, Any] | None = None):
        self.provider.launch(self, launch_kwargs)

    def kill(self, launch_kwargs: dict[str, Any] | None = None):
        self.provider.kill(self, launch_kwargs)

    def dump_state(self):
        state_keys = [
            "model",
            "model_type",
            "cache",
            "num_retries",
            "finetuning_model",
            "launch_kwargs",
            "train_kwargs",
        ]
        return {key: getattr(self, key) for key in state_keys} | self.kwargs

    def _check_truncation(self, results):
        
        if self.model_type != "responses" and any(
            c.finish_reason == "length" for c in results["choices"]
        ):
            logger.warning(
                f"LM response was truncated due to exceeding max_tokens={self.kwargs['max_tokens']}. "
                "You can inspect the latest LM interactions with `dspy.inspect_history()`. "
                "To avoid truncation, consider passing a larger max_tokens when setting up dspy.LM. "
                f"You may also consider increasing the temperature (currently {self.kwargs['temperature']}) "
                " if the reason for truncation is repetition."
            )

    def infer_provider(self) -> HyperLocalLocalProvider:
        if "::" in self.model:
            raise NotImplementedError(
                "Logic for handling GGUF models is not implemented yet."
            )
        else:
            return UnslothLocalProvider



import logging
from typing import TYPE_CHECKING, Any, Callable, Literal

from litellm import ContextWindowExceededError

import dspy
from dspy.adapters.types.tool import Tool
from dspy.primitives.module import Module
from dspy.signatures.signature import ensure_signature

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from dspy.signatures.signature import Signature


class HyperLocalReAct(Module):
    def __init__(self, signature: type["Signature"], tools: list[Callable], max_iters: int = 10):
        """
        ReAct stands for "Reasoning and Acting," a popular paradigm for building tool-using agents.
        In this approach, the language model is iteratively provided with a list of tools and has
        to reason about the current situation. The model decides whether to call a tool to gather more
        information or to finish the task based on its reasoning process. The DSPy version of ReAct is
        generalized to work over any signature, thanks to signature polymorphism. This is an adaptation of that
        version, with instructions and data formatting modified for compatibility/increased performance with local models.

        Args:
            signature: The signature of the module, which defines the input and output of the react module.
            tools (list[Callable]): A list of functions, callable objects, or `dspy.Tool` instances.
            max_iters (Optional[int]): The maximum number of iterations to run. Defaults to 10.

        Example:

        ```python
        def get_weather(city: str) -> str:
            return f"The weather in {city} is sunny."

        react = dspy.ReAct(signature="question->answer", tools=[get_weather])
        pred = react(question="What is the weather in Tokyo?")
        ```
        """
        super().__init__()
        self.signature = signature = ensure_signature(signature)
        self.max_iters = max_iters

        tools = [t if isinstance(t, Tool) else Tool(t) for t in tools]
        tools = {tool.name: tool for tool in tools}

        inputs = ", ".join([f"`{k}`" for k in signature.input_fields.keys()])
        outputs = ", ".join([f"`{k}`" for k in signature.output_fields.keys()])
        instr = [f"{signature.instructions}\n"] if signature.instructions else []

        instr.extend(
            [
                f"You are an Agent. In each episode, you will be given the fields {inputs} as input. And you can see your past trajectory so far.",
                f"Your goal is to use one or more of the supplied tools to collect any necessary information for producing {outputs}, which you must do as soon as you know the answer.\n",
                "To do this, you will interleave next_thought, next_tool_name, and next_tool_args in each turn, and also when finishing the task. Leave no field blank. Critically, when finishing the task, you must explicitly state the answer in your final next_thought.\n",
                "After each tool call, you receive a resulting observation, which gets appended to your trajectory. If you come up with nothing, repeat the output of your tools in the trajectory.\n",
                "When writing next_thought, you may reason about the current situation and plan for future steps.",
                "When selecting the next_tool_name and its next_tool_args, the tool must be one of:\n",
            ]
        )

        tools["finish"] = Tool(
            func=lambda: "Completed.",
            name="finish",
            desc=f"Marks the task as complete. That is, signals that all information for producing the outputs, i.e. {outputs}, are now available to be extracted. It is critical that the answer is explicitly stated while using this tool.",
            args={},
        )

        for idx, tool in enumerate(tools.values()):
            instr.append(f"({idx + 1}) {tool}")
        instr.append("When providing `next_tool_args`, the value inside the field must be in JSON format")

        react_signature = (
            dspy.Signature({**signature.input_fields}, "\n".join(instr))
            .append("trajectory", dspy.InputField(), type_=str)
            .append("next_thought", dspy.OutputField(), type_=str)
            .append("next_tool_name", dspy.OutputField(), type_=Literal[tuple(tools.keys())])
            .append("next_tool_args", dspy.OutputField(), type_=dict[str, Any])
        )

        print(signature.instructions)
        fallback_signature = dspy.Signature(
            {**signature.input_fields, **signature.output_fields},
            "Given the field trajectory, and " + signature.instructions ,
        ).append("trajectory", dspy.InputField(), type_=str)

        self.tools = tools
        self.react = dspy.Predict(react_signature)
        print("REACT SIGNATURE", react_signature)
        print("FALLBACK SIGNATURE", fallback_signature)
        self.extract = dspy.ChainOfThought(fallback_signature)

    def _format_trajectory(self, trajectory: dict[str, Any]):
        adapter = dspy.settings.adapter or dspy.ChatAdapter()
        trajectory_signature = dspy.Signature(f"{', '.join(trajectory.keys())} -> x")
        return adapter.format_user_message_content(trajectory_signature, trajectory)

    def forward(self, **input_args):
        trajectory = {}
        max_iters = input_args.pop("max_iters", self.max_iters)
        for idx in range(max_iters):
            try:
                pred = self._call_with_potential_trajectory_truncation(self.react, trajectory, **input_args)
            except ValueError as err:
                logger.warning(f"Ending the trajectory: Agent failed to select a valid tool: {_fmt_exc(err)}")
                break

            trajectory[f"thought_{idx}"] = pred.next_thought
            trajectory[f"tool_name_{idx}"] = pred.next_tool_name
            trajectory[f"tool_args_{idx}"] = pred.next_tool_args

            try:
                trajectory[f"observation_{idx}"] = self.tools[pred.next_tool_name](**pred.next_tool_args)
            except Exception as err:
                trajectory[f"observation_{idx}"] = f"Execution error in {pred.next_tool_name}: {_fmt_exc(err)}"

            if pred.next_tool_name == "finish":
                break

        extract = self._call_with_potential_trajectory_truncation(self.extract, trajectory, **input_args)
        print("EXTRACT", extract)
        return dspy.Prediction(trajectory=trajectory, **extract)

    async def aforward(self, **input_args):
        trajectory = {}
        max_iters = input_args.pop("max_iters", self.max_iters)
        for idx in range(max_iters):
            try:
                pred = await self._async_call_with_potential_trajectory_truncation(self.react, trajectory, **input_args)
            except ValueError as err:
                logger.warning(f"Ending the trajectory: Agent failed to select a valid tool: {_fmt_exc(err)}")
                break

            trajectory[f"thought_{idx}"] = pred.next_thought
            trajectory[f"tool_name_{idx}"] = pred.next_tool_name
            trajectory[f"tool_args_{idx}"] = pred.next_tool_args

            try:
                trajectory[f"observation_{idx}"] = await self.tools[pred.next_tool_name].acall(**pred.next_tool_args)
            except Exception as err:
                trajectory[f"observation_{idx}"] = f"Execution error in {pred.next_tool_name}: {_fmt_exc(err)}"

            if pred.next_tool_name == "finish":
                break

        extract = await self._async_call_with_potential_trajectory_truncation(self.extract, trajectory, **input_args)
        return dspy.Prediction(trajectory=trajectory, **extract)

    def _call_with_potential_trajectory_truncation(self, module, trajectory, **input_args):
        for _ in range(3):
            try:
                return module(
                    **input_args,
                    trajectory=self._format_trajectory(trajectory),
                )
            except ContextWindowExceededError:
                logger.warning("Trajectory exceeded the context window, truncating the oldest tool call information.")
                trajectory = self.truncate_trajectory(trajectory)

    async def _async_call_with_potential_trajectory_truncation(self, module, trajectory, **input_args):
        for _ in range(3):
            try:
                return await module.acall(
                    **input_args,
                    trajectory=self._format_trajectory(trajectory),
                )
            except ContextWindowExceededError:
                logger.warning("Trajectory exceeded the context window, truncating the oldest tool call information.")
                trajectory = self.truncate_trajectory(trajectory)

    def truncate_trajectory(self, trajectory):
        """Truncates the trajectory so that it fits in the context window.

        Users can override this method to implement their own truncation logic.
        """
        keys = list(trajectory.keys())
        if len(keys) < 4:
            # Every tool call has 4 keys: thought, tool_name, tool_args, and observation.
            raise ValueError(
                "The trajectory is too long so your prompt exceeded the context window, but the trajectory cannot be "
                "truncated because it only has one tool call."
            )

        for key in keys[:4]:
            trajectory.pop(key)

        return trajectory


def _fmt_exc(err: BaseException, *, limit: int = 5) -> str:
    """
    Return a one-string traceback summary.
    * `limit` - how many stack frames to keep (from the innermost outwards).
    """

    import traceback

    return "\n" + "".join(traceback.format_exception(type(err), err, err.__traceback__, limit=limit)).strip()


