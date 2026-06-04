import graphviz
import os

# Create Digraph object
dot = graphviz.Digraph(comment='FedKDL Architecture', format='png')
dot.attr(rankdir='TB', size='10,8', dpi='300', nodesep='0.8', ranksep='1.2')
dot.attr('node', shape='box', style='rounded,filled', fontname='Arial', fontsize='12')
dot.attr('edge', fontname='Arial', fontsize='10')

# Tier 3: Surface Gateway
with dot.subgraph(name='cluster_tier3') as c:
    c.attr(label='Tier 3: Surface Gateway (Macro Tier - Surface)', style='dashed', color='blue', bgcolor='#E6F2FF')
    c.node('GW', 'Global Aggregator\n(FedAvg for Student Model)', fillcolor='#99CCFF')
    c.node('KD', 'Knowledge Distillation\n(Teacher -> Student)', fillcolor='#99CCFF')
    c.edge('GW', 'KD', label='  Student Init')

# Tier 2: Relays
with dot.subgraph(name='cluster_tier2') as c:
    c.attr(label='Tier 2: Relays (Intermediate Tier - Mid-water)', style='dashed', color='green', bgcolor='#E6FFE6')
    
    # Relay 1
    c.node('R1_D', 'Dequantization (FP32)', fillcolor='#99FF99')
    c.node('R1_A', 'SVD-LoRA Aggregation', fillcolor='#99FF99')
    c.node('R1_C', 'Neighborhood Cooperation\n(Model Mixing)', fillcolor='#99FF99')
    c.node('R1_Q', 'Quantization (INT8)', fillcolor='#99FF99')
    c.edge('R1_D', 'R1_A')
    c.edge('R1_A', 'R1_C')
    c.edge('R1_C', 'R1_Q')

    # Relay 2
    c.node('R2_D', 'Dequantization (FP32)', fillcolor='#99FF99')
    c.node('R2_A', 'SVD-LoRA Aggregation', fillcolor='#99FF99')
    c.node('R2_C', 'Neighborhood Cooperation\n(Model Mixing)', fillcolor='#99FF99')
    c.node('R2_Q', 'Quantization (INT8)', fillcolor='#99FF99')
    c.edge('R2_D', 'R2_A')
    c.edge('R2_A', 'R2_C')
    c.edge('R2_C', 'R2_Q')
    
    # Cooperation edge
    c.edge('R1_C', 'R2_C', dir='both', color='darkgreen', style='dashed', label='  Peer Exchange')

# Tier 1: AUVs
with dot.subgraph(name='cluster_tier1') as c:
    c.attr(label='Tier 1: AUVs (Micro Tier - Deep-water)', style='dashed', color='orange', bgcolor='#FFF2E6')
    
    # AUV 1
    c.node('A1_D', 'Local Dataset', shape='cylinder', fillcolor='#FFCC99')
    c.node('A1_L', 'Local Fine-tuning\n(FlexLoRA)', fillcolor='#FFCC99')
    c.node('A1_Q', 'Quantization (INT8)', fillcolor='#FFCC99')
    c.edge('A1_D', 'A1_L')
    c.edge('A1_L', 'A1_Q')
    
    # AUV 2
    c.node('A2_D', 'Local Dataset', shape='cylinder', fillcolor='#FFCC99')
    c.node('A2_L', 'Local Fine-tuning\n(FlexLoRA)', fillcolor='#FFCC99')
    c.node('A2_Q', 'Quantization (INT8)', fillcolor='#FFCC99')
    c.edge('A2_D', 'A2_L')
    c.edge('A2_L', 'A2_Q')
    
    # AUV 3
    c.node('A3_D', 'Local Dataset', shape='cylinder', fillcolor='#FFCC99')
    c.node('A3_L', 'Local Fine-tuning\n(FlexLoRA)', fillcolor='#FFCC99')
    c.node('A3_Q', 'Quantization (INT8)', fillcolor='#FFCC99')
    c.edge('A3_D', 'A3_L')
    c.edge('A3_L', 'A3_Q')

# Cross-tier connections
# AUV to Relay (Uplink)
dot.edge('A1_Q', 'R1_D', color='red', label=' Uplink INT8')
dot.edge('A2_Q', 'R1_D', color='red')
dot.edge('A3_Q', 'R2_D', color='red', label=' Uplink INT8')

# Relay to Gateway (Uplink)
dot.edge('R1_Q', 'GW', color='red', label=' Uplink INT8')
dot.edge('R2_Q', 'GW', color='red')

# Gateway to AUVs (Downlink / Broadcast)
# We add invisible nodes to route broadcast lines nicely or just direct lines
dot.edge('KD', 'A1_L', color='blue', style='dotted', label=' Broadcast (Downlink)')
dot.edge('KD', 'A2_L', color='blue', style='dotted')
dot.edge('KD', 'A3_L', color='blue', style='dotted')

# Render the diagram
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'architecture_diagram')
dot.render(output_path, cleanup=True)
print(f"Diagram saved to {output_path}.png")
