FROM public.ecr.aws/lambda/python:3.12-x86_64

COPY . ${LAMBDA_TASK_ROOT}
COPY ./requirements.txt ${LAMBDA_TASK_ROOT}

WORKDIR ${LAMBDA_TASK_ROOT}

RUN pip install -r requirements.txt

CMD [ "app.handler" ]
