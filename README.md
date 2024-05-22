# **description:**

This is a project for simulating LLM performance on wafer scale AI chips and GPUs.

forward dataflow with access type： send/recv with sram or load /store with dram

| dataflow          | input       | weight      | output      | rest_data        |  |
| ----------------- | ----------- | ----------- | ----------- | ---------------- | - |
| weight stream     | none        | send 1/none | none        | none             |  |
| activation stream | send 1/none | none        | send 1/none | load 1           |  |
| input stationary  | load 1      | load a      | load b      | load c           |  |
| weight stationary | load a      | load 1      | store b     | store and load c |  |
