# README (cutter.py)

## Usecase : 
This program is used to extract labels from YoloV5 to a second program as independent images. 

## Coordinates schema : 
```                                                                  
 Origin (0, 0)                                                       
       │                                                             
        ─ ┐◀─────────────────────WIDTH────────────────────▶          
          ●────────────────────────────────────────────────┐ ▲       
          │                                                │ │       
          │              ┌───┐                             │ │       
          │              │top│                             │ │       
          │              └───┘        center               │ │       
          │                │   ┌ ─ ─  (x, y)               │ │       
          │                ▼                               │ │       
          │         ┌──────────┼──┐                        │ │       
          │         │             │                        │ │       
          │         │          │  │                        │ │       
          │ ┌────┐  │      .      │  ┌────┐                │ │       
          │ │left│─▶│     (●)─ ┘  │◀─│righ│                │ HEIGHT  
          │ └────┘  │      '      │  └────┘                │ │       
          │         │             │                        │ │       
          │         │             │                        │ │       
          │         └─────────────┘◀ ─ ┐                   │ │       
          │                ▲                               │ │       
          │                │           │                   │ │       
          │            ┌──────┐                            │ │       
          │            │bottom│        │                   │ │       
          │            └──────┘                            │ │       
          │                            │                   │ │       
          │                                                │ │       
          │                            │                   │ │       
          └────────────────────────────────────────────────┘ ▼       
                                       │                             
                                                                     
                                       └ ─ ─ LABEL                   
```

## Installation guide

### Virtual environment
To use this program we encourage you to use a virtual environment in python. 
To create a `venv` use : (only first use)
```
python3 -m venv env (only first use)
```
activate the `venv` :
```
source env/bin/activate
```
install requirements : 
```
pip install -r requirements.txt
```
activate the `venv` :
```
deactivate
```

### .ENV config file
.ENV file is used to store configuration data for the program. 
It's mendatory to have a `DATAPATH` in this file (direct path, from the root to the main foder of the [dataset git](https://github.com/Anay21110/BloodCell-Detection-Datatset)):
```
DATAPATH='/Direct/Path/To/GitRepo/BloodCell-Detection-Datatset'
```

### RUN the program
To run the program in venv : 
```
(env) python3 cutterTest.py
```