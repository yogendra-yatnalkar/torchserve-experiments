import requests
import random
from locust import HttpUser, task, between, constant

class QuickstartUser(HttpUser):
    wait_time = constant(1)

    @task
    def index_page(self):
        #res = requests.post("http://localhost:8080/predictions/mnist", 
        #                files={'data': open('test_data/0.png', 'rb')})
        # self.client.get("/hello")
        # self.client.get("/world")
        
        files = {"data": open("./0.png", "rb")}
        self.client.post("/predictions/vit_l_16", files=files)