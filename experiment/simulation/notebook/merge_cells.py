
import json
import re

file_path = '/home/shabd/programming/python/dei_ml/vlab/git_repos/exp-activation-optimization-dei/experiment/simulation/notebook/experiments_data_30eps.js'

with open(file_path, 'r') as f:
    content = f.read()

# Strip JS assignment to get JSON
json_str = content.replace('const EXPERIMENTS_DATA = ', '').strip().rstrip(';')

try:
    data = json.loads(json_str)
except json.JSONDecodeError as e:
    print(f"Error parsing JSON: {e}")
    # Fallback/Debug: print start/end to see what's wrong
    print(f"Start: {json_str[:100]}")
    print(f"End: {json_str[-100:]}")
    exit(1)

count = 0

for exp in data:
    cells = exp.get('cells', [])
    i = 0
    while i < len(cells) - 1:
        curr_cell = cells[i]
        next_cell = cells[i+1]
        
        # Check if current cell is the parameters definition
        is_param_cell = False
        if isinstance(curr_cell.get('source'), str):
             if "# Parameters" in curr_cell['source'] and "ACTIVATION =" in curr_cell['source']:
                 is_param_cell = True
        elif isinstance(curr_cell.get('source'), list):
             source_text = "".join(curr_cell['source'])
             if "# Parameters" in source_text and "ACTIVATION =" in source_text:
                 is_param_cell = True
        
        # Check if next cell is the print configuration
        is_print_cell = False
        if isinstance(next_cell.get('source'), str):
            if 'print(f"\\nExperiment Configuration:")' in next_cell['source']:
                is_print_cell = True
        elif isinstance(next_cell.get('source'), list):
            source_text = "".join(next_cell['source'])
            if 'print(f"\\nExperiment Configuration:")' in source_text:
                is_print_cell = True
                
        if is_param_cell and is_print_cell:
            # Merge
            # Combine source
            src1 = curr_cell['source']
            src2 = next_cell['source']
            
            # Normalize to strings if lists
            if isinstance(src1, list): src1 = "".join(src1)
            if isinstance(src2, list): src2 = "".join(src2)
            
            merged_source = src1 + "\n\n" + src2
            
            # Combine outputs
            out1 = curr_cell.get('outputs', [])
            out2 = next_cell.get('outputs', [])
            merged_outputs = out1 + out2
            
            # Update current cell
            curr_cell['source'] = merged_source
            curr_cell['outputs'] = merged_outputs
            
            # Remove next cell
            cells.pop(i+1)
            
            count += 1
            # Don't increment i, as we need to check the current position again (though unlikely to double merge here)
        else:
            i += 1

print(f"Merged {count} cell pairs.")

# Serialize back to JS
new_content = 'const EXPERIMENTS_DATA = ' + json.dumps(data, indent=2) + ';'

with open(file_path, 'w') as f:
    f.write(new_content)
