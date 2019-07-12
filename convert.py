import uff
import sys

if len(sys.argv) != 3:
    print('Usage:', sys.argv[0], ' /path/to/model.pb output_node_name')
    exit()

print('Converting...')
filename = sys.argv[1]
output_filename = filename[:filename.rfind('.')]  + '.uff'
output_node = sys.argv[2]
trt_graph = uff.from_tensorflow_frozen_model(filename, output_nodes=[output_node])
print('Done')
print('Writing to disk...')
with open(output_filename, 'wb') as f:
    f.write(trt_graph)
print('Done')
