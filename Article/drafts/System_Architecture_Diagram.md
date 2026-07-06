# System Architecture Diagram

This Mermaid diagram illustrates the end-to-end pipeline of the DIKD AKI prediction system, from raw EHR data ingestion, through the Differential Privacy mechanism, into the Agentic LLM framework.

```mermaid
flowchart TD
    %% Define Styling
    classDef ehr fill:#e1f5fe,stroke:#039be5,stroke-width:2px;
    classDef dp fill:#f3e5f5,stroke:#8e24aa,stroke-width:2px;
    classDef agent fill:#e8f5e9,stroke:#43a047,stroke-width:2px;
    classDef tool fill:#fff3e0,stroke:#fb8c00,stroke-width:2px;
    classDef output fill:#ffebee,stroke:#e53935,stroke-width:2px;

    %% Data Source
    EHR[(Raw ICU EHR Data<br/>Demographics, SCr, MAP, Meds)]:::ehr
    
    %% DP Module
    subgraph PP [Privacy Preservation]
        DP_Noise[Laplace Noise Injection<br/>&epsilon; Allocation]:::dp
        SynthGen[Differentially Private<br/>Synthetic Data Generator]:::dp
    end
    
    %% Agentic Framework
    subgraph Agent [Agentic LLM Sentinel]
        Gemma[Gemma-4 12B<br/>Clinical Reasoner]:::agent
        Prompt[System Prompt:<br/>Safety Sentinel Guidelines]:::agent
        
        %% Tools
        TimesFM[TimesFM 2.5<br/>48h SCr/MAP Forecaster]:::tool
        KDIGO[KDIGO Rule Engine<br/>Stage 1-3 Definitions]:::tool
    end
    
    %% Final Output
    RiskLabel(["Imminent AKI Risk Label<br/>[AKI_STAGE_1+] or [NORMAL]"]):::output
    ClinNote[Clinical Synthesis Note<br/>Explainable Rationale]:::output

    %% Flow
    EHR ---> PP
    DP_Noise --> SynthGen
    SynthGen --->|Privacy-Preserving<br/>Patient Trajectories| Gemma
    Prompt --> Gemma
    
    Gemma <-->|Queries Trajectory| TimesFM
    Gemma <-->|Checks Criteria| KDIGO
    
    Gemma --> RiskLabel
    Gemma --> ClinNote
```

## How to use this diagram in the manuscript:
If the target journal accepts Mermaid diagrams directly (some online HTML platforms do), you can paste the block above. If the journal requires a standard image format (PDF/PNG), you can use the [Mermaid Live Editor](https://mermaid.live/) to copy-paste this code and export it as a high-resolution SVG or PNG for publication.
