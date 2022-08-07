# Political Bias Detector - Server

### Requirements
* Docker

### Instructions
1. Place the desired model inside the "server" folder (parent of this README)
2. Build docker image:
`sudo docker image build -t political-bias-detector . [OPTIONAL: --build-arg model_name=<YOUR_MODEL_NAME>]`
3. Run server:
`sudo docker container run -it -p 5000:5000 political-bias-detector`

### Extra
* Clean up unused docker volumes:
`sudo docker system prune -a --volumes`
