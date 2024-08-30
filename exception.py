
SERVING_ERROR_CODE=500
PARAM_ERROR_CODE=400
COMPLETED_CODE=200


class OpenapiProcessingException(Exception):
    def __init__(self, message, code=SERVING_ERROR_CODE):
        self.code = code if code in [COMPLETED_CODE, PARAM_ERROR_CODE, SERVING_ERROR_CODE] else SERVING_ERROR_CODE
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"[{self.code}] {self.message}"