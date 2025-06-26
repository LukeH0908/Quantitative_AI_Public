Dev: Qingjian Shi
If you have any questions you can email me at aqj.shi@gmail.com


You need: 
1. TWS api client

1a. you need to open the port of the tws client 7496 on the application

2. subscription to Alpha Vantage API client with real-time data stream
2a. you must configure your api to allow for real time data stream trhough AlphaX terminal 

rename productiuon.env to ".env" and add your keys.


Teminal (ALL): 

python -m venv venv


Teminal ( Linux/Mac ): 
source venv/bin/activate

Teminal ( Windows Powershell ): 
venv/scripts/activate.ps1 

pip install -r requirements.txt



Run the program:
python exe.py exe_params.json


after that, the program takes care of itself, it automates everything, however some bugs might leave a couple positions unclosed.

I take no responsibility on the profitability/negligence on the losses/gains incurred from software, follow updates on blendingwaves.com

