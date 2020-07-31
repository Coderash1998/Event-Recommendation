import spacy
import random
import devents as de

TRAIN_DATA = [('jobs available for machine learning engineers\r', {'entities': [(0, 4, 'Jobs')]}), ('courses on web development available\r', {'entities': [(0, 7, 'Courses')]}), ('college will conduct a coding competiton\r', {'entities': [(30, 40, 'Competitions')]}), ('welcome to the D23 expo on AI\r', {'entities': [(19, 23, 'Expos')]}), ('get a certification on Iot today\r', {'entities': [(6, 19, 'Certifications')]}), ('a webinar on higher studies will be held\r', {'entities': [(2, 9, 'Webinars')]}), ('tedx on management will be held \r', {'entities': [(0, 4, 'Talks')]}), ('avail 20% off on higher education by attending this fest\r', {'entities': [(52, 56, 'Fests')]}), ('get a chance to attend the talk show by Bill Gates on cloud computing\r', {'entities': [(27, 36, 'Talks')]}), ('job opening for python developer available\r', {'entities': [(0, 3, 'Jobs')]}), ('get a chance to do an internship on networking\r', {'entities': [(22, 32, 'Internships')]}), ('hackathon on blockchain will be held online\r', {'entities': [(0, 9, 'Hackathons')]}), ('finance is an important aspect to work\r', {'entities': [(34, 38, 'Jobs')]}), ('interns required for system security\r', {'entities': [(0, 7, 'Internships')]}), ('codeathon on mobile applications will be held\r', {'entities': [(0, 9, 'Hackathons')]}), ('workshop on java will be held tomorrow\r', {'entities': [(0, 8, 'Workshops')]}), ('seminar on software architecture will be held\r', {'entities': [(0, 7, 'Seminars')]}), ('get hands on training on c \r', {'entities': [(13, 21, 'Trainings')]}), ('do a course on c++ as it is very helpful\r', {'entities': [(5, 11, 'Courses')]}), ('javascript is essential for this internship\r', {'entities': [(33, 43, 'Internships')]}), ('10% off on data science course\r', {'entities': [(24, 30, 'Courses')]}), ('get a system administration certification from purplehat today\r', {'entities': [(28, 41, 'Certifications')]}), ('systems analyst / systems engineer jobs are open\r', {'entities': [(35, 39, 'Jobs')]}), ('business systems analyst are provided with trainings\r', {'entities': [(43, 52, 'Trainings')]}), ('crm business analyst wil host a webinar\r', {'entities': [(32, 39, 'Webinars')]}), ('software systems engineer do good in hackathons\r', {'entities': [(37, 47, 'Hackathons')]}), ('seminar on solution architect will be held\r', {'entities': [(0, 7, 'Seminars')]}), ('e-commerce analyst are good interns\r', {'entities': [(28, 35, 'Internships')]}), ('erp business analyst will conduct a workshop\r', {'entities': [(36, 44, 'Workshops')]}), ('product designers are essential in expos\r', {'entities': [(35, 40, 'Expos')]}), ('human resources are present in every job\r', {'entities': [(37, 40, 'Jobs')]}), ('graphics designing is a good job\r', {'entities': [(29, 32, 'Jobs')]}), ('seo specialist would be training us\r', {'entities': [(24, 32, 'Trainings')]}), ('market research analyst will give us a talk today\r', {'entities': [(39, 43, 'Talks')]}), ('architectural technologist is an interesting internship\r', {'entities': [(45, 55, 'Internships')]}), ('every job has business analyst who gives good insights\r', {'entities': [(6, 9, 'Jobs')]}), ('database administration certification is available\r', {'entities': [(24, 37, 'Certifications')]}), ('electronics enginner are required for this hackathon\r', {'entities': [(43, 52, 'Hackathons')]}), ('sql is an important course\r', {'entities': [(20, 26, 'Courses')]}), ('every job should have an insurance\r', {'entities': [(6, 9, 'Jobs')]}), ('there is a webinar on telecom industry\r', {'entities': [(11, 18, 'Webinars')]}), ('administration assistant required for internship\r', {'entities': [(38, 48, 'Internships')]}), ('social media advertising for this expo is required\r', {'entities': [(34, 38, 'Expos')]}), ('seminar on technical engineer is at 6\r', {'entities': [(0, 7, 'Seminars')]}), ('pre-sales engineer is required for this workshop\r', {'entities': [(40, 48, 'Workshops')]}), ('in this seminar we will learn about portal administrator\r', {'entities': [(8, 15, 'Seminars')]}), ('a good programmer analyst will give us a tedtalk\r', {'entities': [(41, 48, 'Talks')]}), ('network analyst will be attending the webinar\r', {'entities': [(38, 45, 'Webinars')]}), ('network engineer is a good profession\r', {'entities': [(27, 37, 'Jobs')]}), ('get a chance to obtain certification on wireless engineer\r', {'entities': [(23, 36, 'Certifications')]}), ('business continuity analyst will be inaugurating the fest\r', {'entities': [(53, 57, 'Fests')]}), ('erp technical analyst will conduct a workshop\r', {'entities': [(37, 45, 'Workshops')]}), ('erp functional analyst plans the seminar\r', {'entities': [(33, 40, 'Seminars')]}), ('database administrator required for the job\r', {'entities': [(40, 43, 'Jobs')]}), ('software developer will be the judge for this expo\r', {'entities': [(46, 50, 'Expos')]}), ('telecommunication manager will train us\r', {'entities': [(31, 36, 'Trainings')]}), ('the erp technical developer will be conducting this competition\r', {'entities': [(52, 63, 'Competitions')]}), ('network manager is required for this job\r', {'entities': [(37, 40, 'Jobs')]}), ('project manager course avaiable here\r', {'entities': [(16, 22, 'Courses')]}), ('webinar on application development will be held today\r', {'entities': [(0, 7, 'Webinars')]}), ('system security administrator will conduct a training exercise\r', {'entities': [(45, 53, 'Trainings')]}), ('network security engineer is a good job\r', {'entities': [(36, 39, 'Jobs')]}), ('data warehouse developer will be attending the fest\r', {'entities': [(47, 51, 'Fests')]}), ('data analyst is conducting a webinar\r', {'entities': [(29, 36, 'Webinars')]}), ('every database developer has to do this course\r', {'entities': [(40, 46, 'Courses')]}), ('data modeler is a good profession\r', {'entities': [(23, 33, 'Jobs')]}), ('this is a good internship for web developer\r', {'entities': [(15, 25, 'Internships')]}), ('a product manager will guide us in this expo\r', {'entities': [(40, 44, 'Expos')]}), ('data security analyst will give us a talk\r', {'entities': [(37, 41, 'Talks')]}), ('information security analyst will be conducting a webinar\r', {'entities': [(50, 57, 'Webinars')]}), ('every applications developer did this course\r', {'entities': [(38, 44, 'Courses')]}), ('this course on design is good\r', {'entities': [(5, 11, 'Courses')]}), ('ux is important to attend this webinar\r', {'entities': [(31, 38, 'Webinars')]}), ('information technology manager will be conducting the workshop\r', {'entities': [(54, 62, 'Workshops')]}), ('its necessary to do this course for a mobile applications developer\r', {'entities': [(25, 31, 'Courses')]}), ('information technology auditor will be conducting a training session\r', {'entities': [(52, 60, 'Trainings')]}), ('quality assurance analyst is a good job\r', {'entities': [(36, 39, 'Jobs')]}), ('this internship requires database manager skills\r', {'entities': [(5, 15, 'Internships')]}), ('software quality assurance will be conducting a webinar\r', {'entities': [(48, 55, 'Webinars')]}), ('a seminar will be held on data architect\r', {'entities': [(2, 9, 'Seminars')]}), ('data warehouse is an interesting course for future\r', {'entities': [(33, 39, 'Courses')]}), ('become a certified network architect today\r', {'entities': [(9, 18, 'Certifications')]}), ('this job requires skills in react\r', {'entities': [(5, 8, 'Jobs')]}), ('this job requires skills in react native\r', {'entities': [(5, 8, 'Jobs')]}), ('a webinar will be held on php\r', {'entities': [(2, 9, 'Webinars')]}), ('an experienced webdev will be conducting training sessions \r', {'entities': [(41, 49, 'Trainings')]}), ('this internship requires a full stack developer\r', {'entities': [(5, 15, 'Trainings')]}), ('bootstrap is essential for this hackathon\r', {'entities': [(32, 41, 'Hackathons')]}), ('tensorflow is a great topic for this seminar\r', {'entities': [(37, 44, 'Seminars')]}), ('ai is an important course\r', {'entities': [(19, 25, 'Courses')]}), ('neural networks is an interesting topic for this webinar\r', {'entities': [(49, 56, 'Webinars')]}), ('an ethical hacking competition will be held today\r', {'entities': [(19, 30, 'Competitions')]}), ('there will be a talk on phishing\r', {'entities': [(16, 20, 'Talks')]}), ('nodejs is a good course\r', {'entities': [(17, 23, 'Courses')]}), ('this job requires angularjs\r', {'entities': [(5, 8, 'Jobs')]}), ('a webinar on flask will be held\r', {'entities': [(2, 9, 'Webinars')]}), ('hackerrank is a good platform for hackathons\r', {'entities': [(34, 44, 'Hackathons')]}), ('competitive programming is essential skill required for this job\r', {'entities': [(61, 64, 'Jobs')]}), ('icpc is a great competition\r', {'entities': [(16, 27, 'Competitions')]}), ('codechef is a great platform for competitions\r', {'entities': [(33, 45, 'Competitions')]}), ('leetcode is a great platform for competitions\r', {'entities': [(33, 45, 'Competitions')]}), ('raspberry pi is essential for this training\r', {'entities': [(35, 43, 'Trainings')]}), ('a seminar on stock marketing will be held today\r', {'entities': [(2, 9, 'Seminars')]}), ('html is a basic fundamental course required\r', {'entities': [(28, 34, 'Courses')]}), ('this webinar will provide hands on training of css\r', {'entities': [(5, 12, 'Webinars')]}), ('ann is good for certification\r', {'entities': [(16, 29, 'Certifications')]}), ('knowledge of cnn is required for this internship\r', {'entities': [(38, 48, 'Internships')]}), ('rnn is used in this job\r', {'entities': [(20, 23, 'Jobs')]}), ('a workshop on django will be conducted soon\r', {'entities': [(2, 10, 'Workshops')]}), ('an expo on autocad will be held\r', {'entities': [(3, 7, 'Expos')]}), ('swift is an important course\r', {'entities': [(22, 28, 'Courses')]}), ('get certified for android studio today\r', {'entities': [(4, 13, 'Certifications')]}), ('kotlin is required for this job\r', {'entities': [(28, 31, 'Jobs')]}), ('there will be a talk on ios by a great person\r', {'entities': [(16, 20, 'Talks')]}), ('kaggle is conducting a hackathon\r', {'entities': [(23, 32, 'Hackathons')]}), ('tkinter is important for this internship\r', {'entities': [(30, 40, 'Internships')]}), ("cyber security is the theme for this year's fest\r", {'entities': [(44, 48, 'Fests')]}), ('a webinar on malware will be held\r', {'entities': [(2, 9, 'Webinars')]}), ('azure is conducting a seminar\r', {'entities': [(22, 29, 'Seminars')]}), ('aws will be hosting the expo online\r', {'entities': [(24, 28, 'Expos')]}), ('get a job in amazon web services\r', {'entities': [(6, 9, 'Jobs')]}), ('ibm is conducting a hackathon\r', {'entities': [(20, 29, 'Hackathons')]}), ('mba is required for this job\r', {'entities': [(25, 28, 'Jobs')]}), ('a webinar for ms students will be held soon\r', {'entities': [(2, 9, 'Webinars')]}), ('mtech students should attend this expo\r', {'entities': [(34, 38, 'Expos')]}), ('seminar on gre will be held soon\r', {'entities': [(0, 7, 'Seminars')]}), ('this fest will be held for gmat students\r', {'entities': [(5, 9, 'Fests')]}), ('cat students should attend this workshop\r', {'entities': [(32, 40, 'Workshops')]}), ('training session for toefl will start soon\r', {'entities': [(0, 8, 'Trainings')]}), ('gate aspirants should attend this webinar on monday\r', {'entities': [(34, 41, 'Webinars')]}), ('this webinar will throw some light on ielts\r', {'entities': [(5, 12, 'Webinars')]}), ('workshop on kali linux will be held tomorrow\r', {'entities': [(0, 8, 'Workshops')]}), ('big data analytics is required for this internship\r', {'entities': [(40, 50, 'Internships')]}), ('this job requires skills on tableau\r', {'entities': [(5, 8, 'Jobs')]}), ('power bi is an important skill for this job\r', {'entities': [(40, 43, 'Jobs')]}), ('intelligent system is a good course\r', {'entities': [(29, 35, 'Courses')]}), ('a workshop on fuzzy logic will be conducted\r', {'entities': [(2, 10, 'Workshops')]}), ('there will be a talk on fuzzy logic\r', {'entities': [(16, 20, 'Talks')]}), ('a seminar on supervised learning will be held\r', {'entities': [(2, 9, 'Seminars')]}), ('a webinar on unsupervised learning will be conducted\r', {'entities': [(2, 9, 'Webinars')]}), ('interns required for reinforcement learning project\r', {'entities': [(0, 7, 'Internships')]}), ('we will provide training on natural language processing \r', {'entities': [(16, 24, 'Trainings')]}), ('hackathon on nlp will be conducted\r', {'entities': [(0, 9, 'Hackathons')]}), ('this job requires deep learning knowledge\r', {'entities': [(5, 8, 'Jobs')]}), ('there will be a seminar on chatbot\r', {'entities': [(16, 23, 'Seminars')]}), ('free virtual session on iot\r', {'entities': [(5, 20, 'Webinars')]}), ("codechef's november challenge competition", {'entities': [(30, 41, 'Competitions')]}), ('machine learning is an interesting course', {'entities': [(35, 41, 'Courses')]}), ('IIT Bombay is conducting a techfest', {'entities': [(27, 35, 'Fests')]}), ('CCL free ml internship', {'entities': [(12, 22, 'Internships')]}), ('CCL free ml internship', {'entities': [(12, 22, 'Internships')]}), ('placements open for ce/it students\r', {'entities': [(0, 10, 'Jobs')]}), ('interview will start at 11\r', {'entities': [(0, 9, 'Jobs')]}), ('november long contest will begin soon\r', {'entities': [(14, 21, 'Competitions')]}), ('codechef will help u to practice\r', {'entities': [(0, 8, 'Competitions')]}), ('an online session will be held on ai\r', {'entities': [(3, 17, 'Webinar')]}), ('this event will take place in mumbai\r', {'entities': [(5, 10, 'Fests')]}), ('bootcamp for this ibm course will start\r', {'entities': [(0, 8, 'Hackathons'), (22, 28, 'Courses')]}), ('There will be an online drive for final year students', {'entities': [(24, 29, 'Jobs')]})]

def train_spacy(data,iterations):
    TRAIN_DATA = data
    nlp = spacy.blank('en')  # create blank Language class
    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
       

    # add labels
    for _, annotations in TRAIN_DATA:
         for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(iterations):
            print("Statring iteration " + str(itn))
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                nlp.update(
                    [text],  # batch of texts
                    [annotations],  # batch of annotations
                    drop=0.2,  # dropout - make it harder to memorise data
                    sgd=optimizer,  # callable to update weights
                    losses=losses)
            print(losses)
    return nlp


prdnlp = train_spacy(TRAIN_DATA, 100)

# Save our trained Model
modelfile = input("Enter your Model Name: ")
prdnlp.to_disk(modelfile)
x=de.d_events()
#Test your text
test_text = input("Enter your testing text: ")
doc = prdnlp(test_text)
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)