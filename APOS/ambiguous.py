import util

json_list = util.read_json('Job/raw_data/resume.json')
position_scale_dict = {}
position_salary_dict = {}
for i in xrange(len(json_list)):
    jobs = json_list[i]["workExperienceList"]
    for job_id in xrange(len(jobs)):
        job = jobs[job_id]
        if job is not None:
            position_name = job["position_name"]
            scale = job["size"]
            salary = job["salary"]
            if position_name in position_scale_dict:
                position_scale_dict[position_name].add(scale)
            else:
                position_scale_dict.setdefault(position_name, set([scale]))
            if position_name in position_salary_dict:
                position_salary_dict[position_name].add(salary)
            else:
                position_salary_dict.setdefault(position_name, set([salary]))
sc = len([scale for scale in position_scale_dict.values() if len(scale) > 1])
sa = len([salary for salary in position_salary_dict.values() if len(salary) > 1])
print sc, sa, float(sc) / len(position_scale_dict), float(sa) / len(position_salary_dict)

file_path = ['Pos/data/oct27.conll']
file_lines = open(file_path[0]).readlines()
word_list = [line.split('	')[0] for line in file_lines if len(line) > 1]
label_list = [line.split('	')[1] for line in file_lines if len(line) > 1]
word_label_dict = {}
for index, word in enumerate(word_list):
    if word in word_label_dict:
        word_label_dict[word].add(label_list[index])
    else:
        word_label_dict.setdefault(word, set([label_list[index]]))
wl = len([label for label in word_label_dict.values() if len(label) > 1])
print wl, len(word_label_dict), float(wl) / len(word_label_dict)

file_path = ['Pos/data/daily547.conll']
file_lines = open(file_path[0]).readlines()
word_list = [line.split('	')[0] for line in file_lines if len(line) > 1]
label_list = [line.split('	')[1] for line in file_lines if len(line) > 1]
word_label_dict = {}
for index, word in enumerate(word_list):
    if word in word_label_dict:
        word_label_dict[word].add(label_list[index])
    else:
        word_label_dict.setdefault(word, set([label_list[index]]))
wl = len([label for label in word_label_dict.values() if len(label) > 1])
print wl, len(word_label_dict), float(wl) / len(word_label_dict)
