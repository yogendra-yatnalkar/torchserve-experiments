torchserve --stop
rm -rf ./model_store/vit_l_16.mar
torch-model-archiver --model-name vit_l_16 --version 1.0 --serialized-file vit_l_16.pt --handler custom_handler.py
mv vit_l_16.mar ./model_store
# torchserve --start --model-store ./model_store --models vit_l_16=vit_l_16.mar --ts-config config.properties
torchserve --start --ts-config config.properties