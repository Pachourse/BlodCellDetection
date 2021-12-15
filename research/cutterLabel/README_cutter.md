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

