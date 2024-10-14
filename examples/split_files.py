import os

def split_csv(input_file, output_folder, max_size_bytes):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(input_file, 'r', encoding='utf-8') as infile:
        header = infile.readline()
        part_num = 1
        outfile_path = os.path.join(output_folder, f"recipes_data_part_{part_num}.csv")
        outfile = open(outfile_path, 'w', encoding='utf-8')
        outfile.write(header)
        current_size = os.path.getsize(outfile_path)

        for line in infile:
            line_size = len(line.encode('utf-8'))
            if current_size + line_size > max_size_bytes:
                outfile.close()
                part_num += 1
                outfile_path = os.path.join(output_folder, f"recipes_data_part_{part_num}.csv")
                outfile = open(outfile_path, 'w', encoding='utf-8')
                outfile.write(header)
                current_size = os.path.getsize(outfile_path)
            outfile.write(line)
            current_size += line_size

        outfile.close()
    print(f"File splitting completed. Created {part_num} split files.")

if __name__ == "__main__":
    input_csv = r"C:\Users\hshum\OneDrive\Desktop\Python\CafeCorner\parsely_tool\assets\files\recipes_data\recipes_data.csv"
    output_dir = r"C:\Users\hshum\OneDrive\Desktop\Python\CafeCorner\parsely_tool\assets\files\recipes_data\split_files"
    max_file_size = 500 * 1024 * 1024  # 500 MB

    split_csv(input_csv, output_dir, max_file_size)
