## ARCHITECTURE

The CloudServe ticket support system follows a micro-service archiecture. It includes components like the App, Prometheus Client, Prometheus Server and the Grafana. The **App** includes the main endpoints of this architecture. Grafana, Prometheus Client and Server are for visualizations of metrics through dashboard.


## High Level Architecture

The following diagram is a high-level archiecture of all the services in this micro-service architecture.

```mermaid
flowchart LR
  subgraph Client
    UI[ADK Web App]
  end
  subgraph Services
    App[App]
    PromClient[Prometheus Client]
    PromServer[Prometheus Server]
    Grafana[Grafana Client]
  end
  subgraph Storage
    VectorDB[(Vector DB)]
  end
  UI --> App
  App --> PromClient
  PromServer -- polls --> PromClient
  Grafana -- polls --> PromServer
  App --> VectorDB
  App -. response .-> UI
```

## App Service Architecture

The APP Service is the main service of this architecture. It follows an agentic approach. Here Google ADK has been used for orchestration of agents. Moreover, Chroma has been used as the vector database for storing the knowledge base of documents.

```mermaid
flowchart LR
    InputParser[Input Parser]
    subgraph "Ingestion Agent"
        Ingester[Ingester]
        IngesterNote>Normalizes Ticket]
    end
    subgraph "Classifier Agent"
        Classifier[Classifier]
        ClassifierNote>Classifies Tickets by intent, category and urgency using RandomForest Model]
    end
    subgraph "Retriever Agent"
        Retriever[Retriever]
        RetrieverNote>Retrieves relevant docs from ChromaDB vector database]
    end
    subgraph "Router Agent"
        Router[Router]
        RouterNotes>Decides between Auto-respond vs Escalate using Rule-based approach]
    end
    subgraph "Generator Agent"
        Generator[Generator]
        GeneratorNote>Generates response]
    end
    subgraph "Validator Agent"
        Validator[Validator]
        ValidatorNote>Applies guardrails against the response]
    end
    subgraph Storage
        VectorDB[(Vector DB)]
    end
    InputParser --> Ingester
    Ingester --> Classifier
    Classifier --> Retriever
    Retriever --> Router
    Router --> Generator
    Generator --> Validator
    Retriever -. fetch .-> VectorDB
```