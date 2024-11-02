# ChatBot_JMMAPI_AWS_Lambda_Configuration

For running this API from your side you need to configure the serverless CLI in your local console
and run this command after you have configure the **serverless.yml** with your credentials and environment variables.

```python
sls deploy
```

Here you can see a detailed pipeline schematic of the API architecture:

![image](https://github.com/user-attachments/assets/81954a1f-8b5e-4d83-a11d-2d8a5ee93cb7)


Before anything be sure you have access and permissions for ECR, S3, Bedrock and Sagemaker services in AWS, an snapshot
of the different permissions you need to configure this API are here. For any detail about configuration and deployment 
please don't hesitate to contact me [here](juan.mayortorres@ex-staff.unitn.it).

![image](https://github.com/user-attachments/assets/7f3cdbfb-796e-44a1-88f2-5feb006dbcd0)


Here are two FrontEnd examples of this execution.

![image](https://github.com/user-attachments/assets/a4833eeb-bf5a-47e1-8ce2-892ab907e677)

![image](https://github.com/user-attachments/assets/00554651-f9cf-4147-894d-6e7d2dcaa303)

Please in order to execute and interact with the API follow the instructions and sequences described in this explanatory video
[https://drive.google.com/file/d/1WdxZR2jNM9fGjukFNCYyiFuWVK2nICHK/view?usp=sharing](https://drive.google.com/file/d/1WdxZR2jNM9fGjukFNCYyiFuWVK2nICHK/view?usp=sharing)


