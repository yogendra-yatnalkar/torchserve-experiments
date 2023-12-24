import requests
res = requests.post(
    "http://localhost:8080/predictions/vit_l_16", 
    files={"data": open("./0.png", "rb")}
    )

print("Res: ", res.json())