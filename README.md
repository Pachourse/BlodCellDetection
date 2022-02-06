README.md


Student's project at [EPITA](https://www.epita.fr). 

Idea : Use a CNN network to classify BWC after a YOLO-Vs to identify them. 

Global diagram of our implementation 
```
┌─────────────┐       ┌─────────────┐   RBC        ┌─────────────┐                         ┌─────────────┐
│             │       │             │   WBC        │             │                         │             │
│             │       │             │   PLA        │             │   PLA                   │             │
│  DATASET 1  ├──────▶│  YOLO V5s   │───────────┬─▶│   CUTTER    │──────────┬─────────────▶│     CNN     │──┐
│             │       │             │           │  │             │          │              │             │  │
│             │       │             │           │  │             │          ▼              │             │  │
└─────────────┘       └─────────────┘           │  └─────────────┘   ┌────────────┐        └─────────────┘  │
                                                │                    │ /transform │◀────────────────────────┘
                                                │                    └────────────┘      fichier renommé
                                                │                           │
                                                │                           │
                                                └───────────────────────────┘
```

Authors : 
- keith.aroul
- paul.viallet
- paviel.schertzer
- sylvain.keosouk
