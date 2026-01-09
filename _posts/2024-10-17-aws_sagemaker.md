---
title: "Create your own ChatGPT using SageMaker Python SDK V3"
date: 2025-12-06T08:00:30-04:00
categories:
  - Generative AI
  - AWS
classes: wide
excerpt: Tutorial on how to use AWS Sagemaker Python SDK V3 for deploying your own LLM

---

## Introduction
AWS SageMaker Python SDK V3 has introduced more streamlined and unified core classes for ML workflow by replacing previous framework specific classes. This new SDK has a unified “ModelTrainer” class for handling custom containers and data processing, “ModelBuilder” class for model deployment and inference setup. 

I didn't find much resources to deploy custom model in AWS SageMaker using latest python SDK V3, so In this tutorial, I will deploy open source LLM (Qwen3-4B-Instruct-2507) to demonstrate:
1. New deployment process using “ModelBuilder”, including custom dependency and environment variables for deployment containers
2. Custom model loading and inference invocation pipeline inside the containers
3. Schema driven inference for request/response validation

## Prerequisites
Setup your SageMaker studio by creating a domain in Amazon SageMaker AI, then launch a JupyterLab in the SageMaker studio. For this tutorial, minimum instance type for JupyterLab will be enough. However, I will use a GPU instance for deployment endpoint, so be cautious about the incurred costs and always **CLEAN UP** the resources if you are not using them.

## Dependencies Managements
In order to use codes from this tutorial, a recent version of sagemaker and other dependencies is needed. In my case, even starting with the recent distribution of SageMaker in JupyterLab, I was getting older versions. So, I used the following commands in the Jupyter notebook to update my dependencies.

```
%pip install --no-cache-dir -U sagemaker protobuf --quiet
%pip uninstall -y sagemaker sagemaker-core sagemaker-train sagemaker-serve sagemaker-mlops --quiet
%pip install --no-cache-dir sagemaker-core sagemaker-train sagemaker-serve sagemaker-mlops 'sagemaker>=3' --quiet

%pip uninstall -y tensorflow --quiet
%pip install --no-cache-dir tensorflow --quiet
```
## Import Libraries
```
import json
import uuid

from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.serve.spec.inference_spec import InferenceSpec
from sagemaker.serve.builder.schema_builder import SchemaBuilder
from sagemaker.serve.utils.types import ModelServer
from sagemaker.core.resources import EndpointConfig
```

## Custom InferenceSpec
As compared to the previous version of SDK, InferenceSpec allows for customizing model loading and data processing for inference, inside the deployment containers. In the given script, I am inheriting the InferenceSpec into a class to override the load() and invoke() function to suit model loading and data processing pipeline for the “Qwen3-4B-Instruct-2507" model. Here, load() function is one time calling to initialize the tokenizer and the model while starting the deployment container. Then, each time the endpoint gets the requests, invoke() function is executed.

```
class HuggingFaceInferenceSpec(InferenceSpec):
    def __init__(self):
        self.model_name = "Qwen/Qwen3-4B-Instruct-2507"

    def get_model(self):
        return self.model_name
            
    def load(self, model_dir: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model.eval()
        
        return {"model": model, "tokenizer": tokenizer}
            
    def invoke(self, input_object, model):
        import torch
        try:
            tokenizer = model["tokenizer"]
            hf_model = model["model"]
    
            if isinstance(input_object, dict) and "inputs" in input_object:
                messages = input_object["inputs"]
            else:
                messages = [{"role": "user", "content": str(input_object)}]
    
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    
            inputs = tokenizer(text, return_tensors="pt").to(hf_model.device)
    
            with torch.no_grad():
                outputs = hf_model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                )
    
            generated = outputs[0][inputs["input_ids"].shape[-1]:]
            response = tokenizer.decode(generated, skip_special_tokens=True)
    
            return [{"response": response}]

        except Exception as ex:
            return [{"response": f"Error during invocation: {ex}"}]
```

## Schema Builder
I also implement SchemaBuilder to enforce schema aware inference so that our input/output is consistent and is validated. 
```
sample_input = {"inputs": [{"role": "user", "content": "What is AWS Sagemaker?"}]}

sample_output = [{"response": "Amazon SageMaker is a fully managed cloud-based platform provided by AWS "}]
schema_builder = SchemaBuilder(sample_input, sample_output)

print("Schema builder created successfully!")
```

## Model Builder
The parameters are defined for the deployment for HuggingFace model ID, instance type for endpoint, model name and endpoint name for their tracking. Along with this, I am enforcing dependencies and installing them manually into the container as QWEN-3 models depend upon the latest transformer architecture. Similarly, I am using environment variables for transformers such as “HF_MODEL_ID" which downloads and uses the specific model from configured parameters.

```
# Configuration Parameters
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
INSTANCE_TYPE = "ml.g4dn.xlarge"
MODEL_NAME_PREFIX = "hf-v3-qwen3-4b"
ENDPOINT_NAME_PREFIX = "hf-v3-qwen3-4b-endpoint"

dependencies = {
    "auto": False,
    "custom": ["sagemaker>=3.1.1",
               "transformers>=4.57.3", 
               "accelerate",
               "torch>=2.6.0", 
               "cloudpickle>=2.2.1",
               "protobuf"]
}

env_vars = {
    "HF_MODEL_ID": MODEL_ID,
    "HF_TASK": "text-generation",
    "HF_HOME": "/opt/ml/model",
    "TRANSFORMERS_CACHE": "/opt/ml/model",
}

# Generate unique identifiers
unique_id = str(uuid.uuid4())[:8]
model_name = f"{MODEL_NAME_PREFIX}-{unique_id}"
endpoint_name = f"{ENDPOINT_NAME_PREFIX}-{unique_id}"
```

Having parameters, dependencies and environment variables configured, alongside initialized custom InferenceSpec, the next step is to build the model using “ModelBuilder”. Here, as I am using a transformer based model, I am using TORCHSERVER as model server for inference. This will create a model which is like a docker image and can be viewed in the “Models/My models” tab in SageMaker studio. 
```
# Create ModelBuilder
inference_spec = HuggingFaceInferenceSpec()
model_builder = ModelBuilder(
    inference_spec=inference_spec,
    model_server=ModelServer.TORCHSERVE,
    schema_builder=schema_builder,
    instance_type=INSTANCE_TYPE, 
    env_vars=env_vars,
    dependencies=dependencies
)

# Build the model
core_model = model_builder.build(model_name=model_name)
print(f"Model Successfully Created: {core_model.model_name}")
```

![Model Creation Screenshot](/images/blog-images/sagemaker-v3-post/sagemakerv3_model_builder_output.png)

## Model Deployment
In this step, the previously built model will be deployed where containers will be created, InferenceSpec will be executed by pulling the model and loading them into the defined instance type. Then the endpoint will be created and will be live for making API requests.
```
core_endpoint = model_builder.deploy(endpoint_name=endpoint_name)
print(f"Endpoint Successfully Created: {core_endpoint.endpoint_name}")

```
![Model Deployment Screenshot](/images/blog-images/sagemaker-v3-post/sagemakerv3_model_deployment_output.png)

## Testing
Now, when the endpoint is live, you can invoke it and send the requests in the same schema format as defined in the previous step. It will trigger invoke() function and you will get your response from the LLM.
```
test_input_1 = {"inputs": [{"role": "user", "content": "What are major features of AWS Sagemaker?"}]}

result_1 = core_endpoint.invoke(
    body=json.dumps(test_input_1),
    content_type="application/json"
)

response_1 = json.loads(result_1.body.read().decode('utf-8'))
print(f"Conversation Test: {response_1}")
```

![Endpoint Testing Screenshot](/images/blog-images/sagemaker-v3-post/sagemakerv3_final_test_output.png)

## Resource Cleanups
After all the testing is complete and if you are no longer using the endpoint, always clean up the resources using the following script to avoid any further charges.
```
core_endpoint_config = EndpointConfig.get(endpoint_config_name=core_endpoint.endpoint_name)

core_model.delete()
core_endpoint.delete()
core_endpoint_config.delete()

print("All resources successfully deleted!")
```