# Thinking-Bert-Hierarchical-Reasoning-Model
An Experimental attempt to achieve a thinking small Encoder Model based on https://arxiv.org/pdf/2412.13663 (ModernBert) and  https://arxiv.org/abs/2506.21734 (HierarchicalReasoningModel)

## ⚙️ Model Architecture: Information Flow

The `HierarchicalBert` model processes information through a two-level, iterative refinement loop. This diagram shows how data moves through the model's components during its "thinking" process.

```mermaid
graph TD
    subgraph Input Layer
        A[Input IDs & Type IDs] --> B(Embedding & RMSNorm);
    end

    B --> C{Ponder Loop Start};

    subgraph Iteration
        C --> D[Low-Level Processor];
        D --> E[Aggregate Sequence];
        E --> F[High-Level Processor];
        F --> G[Combine States];
        G --> H[Output Head];
        H --> I[Logits for this step];

        F -- "State (zH)" --> J[Q-Head for ACT];
        J --> K[Q-Value for this step];

        G -- "Next States (zL, zH)" --> C;
    end

    subgraph Final Outputs
        I --> L[List of All Logits];
        K --> M[List of All Q-Values];
    end
```

## ⚙️ Model Information Flow: Inference

```mermaid
graph TD
    A["Input: 'The capital of France is <MASK>.'"] --> B["Tokenization"];
    B --> C["Locate Mask Index"];
    C --> D["Pad to Max Length"];
    D --> E["Prepare Tensors"];
    E --> F["model(..., N, T)"];

    subgraph "Model Inference"
        F --> G["Iterative Pondering (N steps)"];
        G --> H["Get Final Logits"];
    end

    H --> I["Extract Logits at Mask Index"];
    I --> J["Find Highest Logit (Argmax)"];
    J --> K["Decode Predicted Token ID"];
    K --> L["Output: 'The capital of France is Paris.'"];
```

## ⚙️ Benchmarks

## ⚙️ Different Sizes models

## ⚙️ Multi token prediction

## ⚙️ Hardware requirements

