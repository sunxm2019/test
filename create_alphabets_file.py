import os


def get_key_string_from_source_txt_file(source_txt_file, code='utf-8-sig'):
	assert os.path.exists(source_txt_file)

	print('reading file: source_txt_file')
	with open(source_txt_file, 'rb') as file:
		txt_list = [part.decode(code, 'ignore').strip() for part in file.readlines()]

	txt_str = ''.join(txt_list[0])
	txt_list = list(set(txt_str))
	return ''.join(txt_list)


if __name__ == '__main__':
	source_txt_file = '.\\source_small.txt'
	output_file = '.\\alphabets.py'

	target_str = get_key_string_from_source_txt_file(source_txt_file)
	assert len(target_str) > 0

	print('key count: ', len(target_str))
	with open(output_file, 'w', encoding='utf-8') as f:
		f.write(r'alphabet = """' + target_str + r'"""' + '\n')
	print('ok')
