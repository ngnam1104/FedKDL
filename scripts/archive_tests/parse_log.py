import re
import csv
import sys
import argparse

def parse_log_file(input_file, output_file):
    # Groups of metrics
    physical_keys = [
        "Round", "loss", "alive", "min_battery", "tau_round_s", "tau_status",
        "tau_a2r", "tau_r2r", "tau_r2g", "tau_comp", "tau_cumul_s",
        "avg_payload_kb", "payload_cumul_kb", "e_total", "e_a2r", "e_r2r",
        "e_r2g", "e_comp", "e_cumul", "lambda_e", "lambda_tau",
        "joint_cost_round", "joint_cost_cumul"
    ]
    
    pre_kd_keys = [
        "pre_kd_mAP50-95", "pre_kd_mAP50", "pre_kd_Prec", "pre_kd_Rec"
    ]
    
    kd_keys = [
        "kd_active", "kd_epochs", "kd_box", "kd_kl", "kd_lora", "kd_weighted"
    ]
    
    post_kd_keys = [
        "mAP50-95", "mAP50", "Prec", "Rec"
    ]
    
    header = physical_keys + pre_kd_keys + kd_keys + post_kd_keys
    
    # Regex to match the round summary line (ignoring prefixes like timestamps)
    round_line_pattern = re.compile(r"Round\s+(\d+)\s+\|(.*)")
    
    rows = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            match = round_line_pattern.search(line.strip())
            if match:
                round_num = match.group(1)
                content = match.group(2)
                
                # Split key-value pairs
                pairs = [p.strip() for p in content.split('|') if p.strip()]
                
                # Dictionary to store current row's data
                row_data = {"Round": round_num}
                
                for pair in pairs:
                    if ':' in pair:
                        # Find the first colon
                        colon_idx = pair.find(':')
                        key = pair[:colon_idx].strip()
                        value = pair[colon_idx+1:].strip()
                        row_data[key] = value
                
                # Extract values in the correct order
                row = []
                for key in header:
                    row.append(row_data.get(key, ""))
                
                rows.append(row)
                
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
        
    print(f"Parsed {len(rows)} rounds and saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse FedKDL raw bash output log into CSV.")
    parser.add_argument("input_log", help="Path to the input raw_bash_output_*.log file")
    parser.add_argument("-o", "--output", default="parsed_metrics.csv", help="Path to output CSV file")
    
    args = parser.parse_args()
    parse_log_file(args.input_log, args.output)
