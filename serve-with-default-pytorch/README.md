# Instruction on how to run end-to-end: 


1. ### Download the model:

```
python download_vit_l_16_weights.py
```

Note: Once the model is downloaded for the first-time, no need to repeat this step again

2. ###  Run the shell script to create the ".mar" file, create a model store and start torchserve 

```
sh update_mar_and_redploy.sh
```

3. ### Check model status: 

```
curl http://localhost:8081/models/vit_l_16
```

4. ### Perform Inference 

```
curl http://127.0.0.1:8080/predictions/vit_l_16 -T 0.png
```

5. ### Start Locust server and perform load testing:

- Type on command line:
  ```
  locust
  ```

- Head over to the following url:
  ```
  localhost:8089
  ```

  Note: If you are using ec2 instance like me:
    - First, add inbound rule to access 8089 port from your ip address
    - Second, head over to the following url: 
  ```
  {public_ip_adress_of_ec2_or_similar_cloud_instance}:8089
  ```
