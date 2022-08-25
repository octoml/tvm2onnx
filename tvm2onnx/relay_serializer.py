"""A utility class for serializing and deserializing Relay models."""
import json
import typing

import tvm

ModelParams = typing.Dict[str, tvm.nd.NDArray]

_FUNC_NAME = "main"
_PARAMS_NAME = "__params__"


class RelaySerializer:
    """A utility class for serializing and deserializing Relay models."""

    @classmethod
    def serialize(cls, model: typing.Tuple[tvm.ir.IRModule, ModelParams]) -> bytes:
        """Converts the given Relay model to bytes.

        :param model: a Relay IR module representing a model.
        :return: bytes representing the given model.
        """
        model = cls._attach_model_params(*model)
        model_json = tvm.ir.save_json(model)
        return bytes(model_json, "utf-8")

    @classmethod
    def deserialize(
        cls, model_bytes: bytes
    ) -> typing.Tuple[tvm.ir.IRModule, ModelParams]:
        """Converts the given bytes to a Relay model.

        :param model_bytes: bytes representing a Relay model.
        :return: the model represented by the given bytes.
        """
        model_json = model_bytes.decode("utf-8")
        json_obj = json.loads(model_json)
        # Add empty attrs line to mitigate deserialization
        # breakage. See OHD-219
        for item in json_obj["nodes"]:
            if item["type_key"] == "IRModule":
                if item["attrs"].get("attrs") is None:
                    item["attrs"].update({"attrs": "0"})
                break
        # Add out_layout to pooling ops to workaround
        # deserialization breakage after TVM PR #9328.
        for item in json_obj["nodes"]:
            if (
                item["type_key"] == "relay.attrs.AdaptivePool1DAttrs"
                or item["type_key"] == "relay.attrs.AdaptivePool2DAttrs"
                or item["type_key"] == "relay.attrs.AdaptivePool3DAttrs"
                or item["type_key"] == "relay.attrs.AvgPool1DAttrs"
                or item["type_key"] == "relay.attrs.AvgPool2DAttrs"
                or item["type_key"] == "relay.attrs.AvgPool3DAttrs"
                or item["type_key"] == "relay.attrs.GlobalPool2DAttrs"
                or item["type_key"] == "relay.attrs.MaxPool1DAttrs"
                or item["type_key"] == "relay.attrs.MaxPool2DAttrs"
                or item["type_key"] == "relay.attrs.MaxPool3DAttrs"
            ):
                if item["attrs"].get("out_layout") is None:
                    item["attrs"].update({"out_layout": item["attrs"].get("layout")})
        model_json = json.dumps(json_obj)
        module = tvm.ir.load_json(model_json)
        return module, cls._get_params(module)

    @classmethod
    def _attach_model_params(
        cls, module: tvm.ir.IRModule, params: ModelParams
    ) -> tvm.ir.IRModule:
        """Attaches the given params to the given module.

        :param module: the module to which the parameters will be attached.
        :param params: the parameters to attach to the module.
        :return: the module with paras attached.
        """
        params = cls._convert_model_params(module, params)
        module[_FUNC_NAME] = module[_FUNC_NAME].with_attr(_PARAMS_NAME, params)
        return module

    @staticmethod
    def _convert_model_params(
        module: tvm.ir.IRModule, params: ModelParams
    ) -> ModelParams:
        """Converts the given module's parameters to be storable with the module.

        :param module: the module whose parameters are to be converted.
        :param params: the parameters to be converted.
        :return: a dict of params able to be attached to the module.
        """
        var_map = {}
        for fn_param in module[_FUNC_NAME].params:
            var_map[fn_param.name_hint] = fn_param

        var_params = {}
        for str_key in params:
            var_key = var_map.get(str_key)
            if var_key:
                var_params[str_key] = tvm.relay.Constant(params[str_key])

        return var_params

    @staticmethod
    def _get_params(module: tvm.ir.IRModule) -> ModelParams:
        """Returns the attached parameters of the given IRModule.

        :return: the parameters of the given module.
        """
        params = module[_FUNC_NAME].attrs[_PARAMS_NAME]
        return {k: v.data for k, v in params.items()}
