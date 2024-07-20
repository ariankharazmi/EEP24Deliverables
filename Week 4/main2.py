import json
import random
from datasets import load_dataset

## WikiText 2 dataset, not used...yet
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
instructions = [ # Initial Instruction Set
    {
        "instruction": "Translate the following sentence to Spanish.",
        "input": "Good morning, how are you?",
        "output": "Buenos días, ¿cómo estás?"
    },
    {
        "instruction": "List the planets in our solar system.",
        "input": "",
        "output": "The planets in our solar system are Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune."
    },
    {
        "instruction": "What is the largest mammal in the world?",
        "input": "",
        "output": "The largest mammal in the world is the blue whale."
    },
    {
        "instruction": "Provide a brief explanation of photosynthesis.",
        "input": "",
        "output": "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with the help of chlorophyll from carbon dioxide and water. It generates oxygen as a byproduct."
    },
    {
        "instruction": "Explain the difference between a laptop and a desktop computer.",
        "input": "",
        "output": "A laptop is a portable computer that integrates all components into one compact unit, allowing for mobility. A desktop computer is larger, consists of separate components such as the monitor, keyboard, and tower, and is typically stationary, offering more power and expandability."
    },
    {
        "instruction": "What are the three branches of the United States government?",
        "input": "",
        "output": "The three branches of the United States government are the Legislative branch (Congress), the Executive branch (President), and the Judicial branch (Supreme Court)."
    },
    {
        "instruction": "Name five different types of clouds.",
        "input": "",
        "output": "Five different types of clouds are: Cumulus, Stratus, Cirrus, Nimbostratus, and Cumulonimbus."
    },
    {
        "instruction": "Summarize the plot of 'To Kill a Mockingbird'.",
        "input": "",
        "output": "'To Kill a Mockingbird' is a novel set in the American South during the 1930s, focusing on the Finch family. The father, Atticus Finch, is a lawyer who defends a black man, Tom Robinson, falsely accused of raping a white woman. The story is narrated by Atticus's young daughter, Scout, and deals with themes of racial injustice, moral growth, and compassion."
    },
    {
        "instruction": "Convert 50 degrees Fahrenheit to Celsius.",
        "input": "",
        "output": "50 degrees Fahrenheit is equivalent to 10 degrees Celsius."
    },
    {
        "instruction": "What is the significance of the year 1776 in American history?",
        "input": "",
        "output": "The year 1776 is significant in American history because it is the year the Declaration of Independence was signed, marking the beginning of the United States as an independent nation."
    },
    {
        "instruction": "Define the term 'ecosystem'.",
        "input": "",
        "output": "An ecosystem is a community of living organisms interacting with one another and their physical environment, including both biotic (living) and abiotic (non-living) components."
    },
    {
        "instruction": "Describe the process of evaporation.",
        "input": "",
        "output": "Evaporation is the process by which water changes from a liquid to a gas or vapor. This occurs when water molecules gain enough energy from heat to break free from the surface and disperse into the air."
    },
    {
        "instruction": "What is the primary function of the circulatory system?",
        "input": "",
        "output": "The primary function of the circulatory system is to transport blood, nutrients, oxygen, carbon dioxide, and hormones throughout the body, providing cells with essential substances and removing waste products."
    },
    {
        "instruction": "Who wrote 'Pride and Prejudice'?",
        "input": "",
        "output": "'Pride and Prejudice' was written by Jane Austen."
    },
    {
        "instruction": "Explain the term 'biodiversity'.",
        "input": "",
        "output": "Biodiversity refers to the variety and variability of life on Earth. It encompasses the diversity of species, genetic variations within species, and the variety of ecosystems."
    },
    {
        "instruction": "Calculate the area of a triangle with a base of 5 cm and a height of 10 cm.",
        "input": "",
        "output": "The area of a triangle with a base of 5 cm and a height of 10 cm is 25 square centimeters. (Area = 1/2 * base * height)"
    },
    {
        "instruction": "What is the chemical formula for water?",
        "input": "",
        "output": "The chemical formula for water is H2O."
    },
    {
        "instruction": "Name three countries in Africa.",
        "input": "",
        "output": "Three countries in Africa are Nigeria, Egypt, and South Africa."
    },
    {
        "instruction": "Identify the author of '1984'.",
        "input": "",
        "output": "'1984' was written by George Orwell."
    },
    {
        "instruction": "Translate the following sentence to French.",
        "input": "I would like a cup of coffee, please.",
        "output": "Je voudrais une tasse de café, s'il vous plaît."
    },
    {
        "instruction": "What is the process of natural selection?",
        "input": "",
        "output": "Natural selection is a process in evolutionary biology where organisms better adapted to their environment tend to survive and produce more offspring. The theory was first formulated by Charles Darwin and is a key mechanism of evolution."
    },
    {
        "instruction": "List four common symptoms of the flu.",
        "input": "",
        "output": "Four common symptoms of the flu are fever, cough, sore throat, and body aches."
    },
    {
        "instruction": "Explain the significance of the Treaty of Versailles.",
        "input": "",
        "output": "The Treaty of Versailles, signed in 1919, formally ended World War I. It imposed heavy reparations and territorial losses on Germany, contributing to economic hardships and political instability, which eventually led to World War II."
    },
    {
        "instruction": "Name three famous classical composers.",
        "input": "",
        "output": "Three famous classical composers are Ludwig van Beethoven, Wolfgang Amadeus Mozart, and Johann Sebastian Bach."
    },
    {
        "instruction": "Describe the water cycle.",
        "input": "",
        "output": "The water cycle describes the continuous movement of water on, above, and below the surface of the Earth. It includes processes such as evaporation, condensation, precipitation, and infiltration."
    },
    {
        "instruction": "What is the smallest prime number?",
        "input": "",
        "output": "The smallest prime number is 2."
    },
    {
        "instruction": "Identify the main causes of climate change.",
        "input": "",
        "output": "The main causes of climate change include greenhouse gas emissions from burning fossil fuels, deforestation, industrial processes, and agricultural activities. These factors increase the concentration of greenhouse gases in the atmosphere, leading to global warming."
    },
    {
        "instruction": "Write a brief biography of Marie Curie.",
        "input": "",
        "output": "Marie Curie was a pioneering physicist and chemist, born in Poland in 1867. She is best known for her research on radioactivity and was the first woman to win a Nobel Prize. She won two Nobel Prizes, one in Physics and one in Chemistry, for her discoveries of radium and polonium. Curie's work laid the foundation for the development of X-ray machines and radiation therapy."
    },
    {
        "instruction": "What is the capital of Japan?",
        "input": "",
        "output": "The capital of Japan is Tokyo."
    },
    {
        "instruction": "Define the term 'quantum computing'.",
        "input": "",
        "output": "Quantum computing is an area of computing that uses principles of quantum mechanics to process information in a fundamentally different way from classical computers. It leverages quantum bits or qubits, which can represent multiple states simultaneously, to solve complex problems more efficiently."
    },
    {
        "instruction": "List the main ingredients in a traditional Caesar salad.",
        "input": "",
        "output": "The main ingredients in a traditional Caesar salad are romaine lettuce, croutons, Parmesan cheese, Caesar dressing, and optionally grilled chicken or anchovies."
    },
    {
        "instruction": "What is the significance of the scientific method?",
        "input": "",
        "output": "The scientific method is a systematic approach to research and problem-solving that involves making observations, forming a hypothesis, conducting experiments, and analyzing results. It is fundamental to scientific inquiry and helps ensure that findings are reliable and valid."
    },
    {
        "instruction": "Describe the function of mitochondria in a cell.",
        "input": "",
        "output": "Mitochondria are known as the powerhouses of the cell. They are responsible for generating most of the cell's supply of adenosine triphosphate (ATP), which is used as a source of chemical energy. Mitochondria also play roles in other processes such as cell signaling and apoptosis."
    },
    {
        "instruction": "Identify the author of 'The Catcher in the Rye'.",
        "input": "",
        "output": "'The Catcher in the Rye' was written by J.D. Salinger."
    },
    {
        "instruction": "What is the primary function of the respiratory system?",
        "input": "",
        "output": "The primary function of the respiratory system is to facilitate the exchange of oxygen and carbon dioxide between the air and the blood. It enables oxygen to enter the bloodstream and carbon dioxide to be expelled from the body."
    },
    {
        "instruction": "Explain the importance of the Magna Carta.",
        "input": "",
        "output": "The Magna Carta, signed in 1215, is a foundational document in the history of democracy. It limited the powers of the king and established certain legal rights for subjects, including the principle that everyone, including the king, was subject to the law."
    },
    {
        "instruction": "List the seven continents of the world.",
        "input": "",
        "output": "The seven continents of the world are Africa, Antarctica, Asia, Europe, North America, Australia, and South America."
    },
    {
        "instruction": "What are the basic principles of democracy?",
        "input": "",
        "output": "The basic principles of democracy include popular sovereignty, where the power rests with the people; rule of law, ensuring that laws apply equally to all citizens; and protection of individual rights and freedoms. Other principles include free and fair elections and the separation of powers among branches of government."
    },
    {
        "instruction": "Provide a brief overview of the Theory of Relativity.",
        "input": "",
        "output": "The Theory of Relativity, formulated by Albert Einstein, consists of two parts: Special Relativity and General Relativity. Special Relativity deals with objects moving at constant high speeds and introduces the concept that the speed of light is constant for all observers. General Relativity extends this to include acceleration and gravity, describing gravity as a curvature of spacetime caused by mass."
    },
    {
        "instruction": "What are the symptoms of diabetes?",
        "input": "",
        "output": "Symptoms of diabetes can include frequent urination, excessive thirst, unexplained weight loss, fatigue, blurred vision, and slow-healing sores. Symptoms may vary depending on whether someone has Type 1 or Type 2 diabetes."
    },
    {
        "instruction": "Name the four seasons of the year.",
        "input": "",
        "output": "The four seasons of the year are spring, summer, autumn (fall), and winter."
    },
    {
        "instruction": "What is the main difference between an asteroid and a comet?",
        "input": "",
        "output": "The main difference between an asteroid and a comet is their composition and appearance. Asteroids are primarily made of rock and metal and are usually found in the asteroid belt between Mars and Jupiter. Comets, on the other hand, are composed of ice, dust, and rocky material and have a characteristic glowing coma and tail when they approach the Sun."
    },
    {
        "instruction": "Describe the purpose of the United Nations.",
        "input": "",
        "output": "The United Nations (UN) is an international organization founded in 1945 to promote peace, security, and cooperation among countries. Its purposes include preventing conflicts, fostering human rights, facilitating international cooperation, and providing humanitarian aid."
    },
    {
        "instruction": "What is the capital city of Canada?",
        "input": "",
        "output": "The capital city of Canada is Ottawa."
    },
    {
        "instruction": "Explain the concept of supply and demand.",
        "input": "",
        "output": "Supply and demand are fundamental concepts in economics. Supply refers to the amount of a product or service available to consumers, while demand refers to how much consumers want the product or service. The interaction between supply and demand determines the price and quantity of goods in the market."
    },
    {
        "instruction": "List five benefits of regular physical exercise.",
        "input": "",
        "output": "Five benefits of regular physical exercise are improved cardiovascular health, increased strength and endurance, better mental health and mood, enhanced weight management, and reduced risk of chronic diseases such as diabetes and hypertension."
    },
    {
        "instruction": "Describe the process of photosynthesis.",
        "input": "",
        "output": "Photosynthesis is the process by which green plants, algae, and some bacteria convert light energy, usually from the sun, into chemical energy in the form of glucose. This process occurs in the chloroplasts of plant cells and involves the absorption of carbon dioxide and water, producing oxygen as a byproduct."
    },
    {
        "instruction": "What is the largest planet in our solar system?",
        "input": "",
        "output": "The largest planet in our solar system is Jupiter."
    },
    {
        "instruction": "Identify the primary colors of light.",
        "input": "",
        "output": "The primary colors of light are red, green, and blue."
    },
    {
        "instruction": "What is the significance of the Pythagorean Theorem?",
        "input": "",
        "output": "The Pythagorean Theorem is a fundamental principle in geometry that states in a right-angled triangle, the square of the length of the hypotenuse (the side opposite the right angle) is equal to the sum of the squares of the lengths of the other two sides. It is expressed as \(a^2 + b^2 = c^2\)."
    },
    {
        "instruction": "Explain the concept of cultural diffusion.",
        "input": "",
        "output": "Cultural diffusion refers to the spread of cultural beliefs, practices, and innovations from one society or cultural group to another. This can occur through various means such as migration, trade, or media and results in the exchange and blending of cultural elements."
    },
    {
        "instruction": "Name the four fundamental forces of nature.",
        "input": "",
        "output": "The four fundamental forces of nature are gravitational force, electromagnetic force, strong nuclear force, and weak nuclear force."
    },
    {
        "instruction": "Describe the role of DNA in living organisms.",
        "input": "",
        "output": "DNA (deoxyribonucleic acid) carries the genetic blueprint for the development, functioning, growth, and reproduction of all living organisms. It contains the instructions for making proteins, which are essential for the structure and function of cells."
    },
    {
        "instruction": "What is the primary function of the liver?",
        "input": "",
        "output": "The primary function of the liver is to process and detoxify substances in the blood, produce bile to aid in digestion, and store nutrients such as glucose and vitamins. It also plays a role in metabolism and the regulation of blood clotting."
    },
    {
        "instruction": "Identify the seven continents by their largest countries.",
        "input": "",
        "output": "1. Africa - Nigeria\n2. Antarctica - No countries\n3. Asia - Russia\n4. Europe - Russia\n5. North America - Canada\n6. Australia - Australia\n7. South America - Brazil"
    },
    {
        "instruction": "What are the main types of economic systems?",
        "input": "",
        "output": "The main types of economic systems are capitalism, socialism, and mixed economies. Capitalism is characterized by private ownership and market-driven economies, socialism involves government control and distribution of resources, and mixed economies combine elements of both capitalism and socialism."
    },
    {
        "instruction": "Explain the difference between a hurricane and a tornado.",
        "input": "",
        "output": "A hurricane is a large, organized system of clouds and thunderstorms featuring a clearly defined circulation pattern and a low-pressure center. It forms over warm ocean waters and can cause widespread damage over a large area. A tornado, on the other hand, is a smaller, more localized column of violently rotating air that extends from a thunderstorm to the ground. Tornadoes are characterized by their funnel shape and are capable of causing intense damage in a narrow path."
    },
    {
        "instruction": "What is the function of the human circulatory system?",
        "input": "",
        "output": "The human circulatory system is responsible for transporting blood, nutrients, gases (such as oxygen and carbon dioxide), and waste products throughout the body. It includes the heart, blood vessels, and blood, and is essential for maintaining homeostasis and overall health."
    },
    {
        "instruction": "Describe the basic principles of Buddhism.",
        "input": "",
        "output": "The basic principles of Buddhism include the Four Noble Truths: 1) Life is suffering (dukkha), 2) Suffering is caused by desire (tanha), 3) Suffering can be overcome, and 4) The path to overcoming suffering is the Eightfold Path. The Eightfold Path consists of right understanding, right intention, right speech, right action, right livelihood, right effort, right mindfulness, and right concentration."
    },
    {
        "instruction": "Name three major works by William Shakespeare.",
        "input": "",
        "output": "Three major works by William Shakespeare are 'Hamlet,' 'Macbeth,' and 'Romeo and Juliet.'"
    },
    {
        "instruction": "What is the main purpose of the World Health Organization (WHO)?",
        "input": "",
        "output": "The main purpose of the World Health Organization (WHO) is to promote global health, provide leadership on international public health issues, and coordinate efforts to combat diseases and health emergencies. The WHO aims to ensure access to essential health services and improve health outcomes worldwide."
    },
    {
        "instruction": "Define the term 'ecosystem'.",
        "input": "",
        "output": "An ecosystem is a biological community of interacting organisms and their physical environment. It includes all living organisms (plants, animals, microorganisms) and non-living components (air, water, soil) that interact as a system."
    },
    {
        "instruction": "What are the key elements of a story plot?",
        "input": "",
        "output": "The key elements of a story plot are exposition (introduction of characters and setting), rising action (development of conflicts), climax (turning point or most intense moment), falling action (events leading to resolution), and resolution (conclusion or outcome of the story)."
    },
    {
        "instruction": "List the major categories of renewable energy sources.",
        "input": "",
        "output": "The major categories of renewable energy sources are solar energy, wind energy, hydroelectric power, geothermal energy, and biomass energy."
    },
    {
        "instruction": "Describe the purpose of the Bill of Rights in the United States.",
        "input": "",
        "output": "The Bill of Rights is the first ten amendments to the United States Constitution and was created to protect individual liberties and rights against government infringement. It includes freedoms such as speech, press, assembly, and religion, as well as protections against unreasonable searches and seizures, and guarantees due process."
    },
    {
        "instruction": "What is the main difference between mitosis and meiosis?",
        "input": "",
        "output": "The main difference between mitosis and meiosis is that mitosis results in two genetically identical daughter cells, while meiosis produces four genetically diverse gametes (sperm or eggs). Mitosis is used for growth and repair, while meiosis is involved in sexual reproduction and the formation of reproductive cells."
    },
    {
        "instruction": "List five countries in South America.",
        "input": "",
        "output": "Five countries in South America are Brazil, Argentina, Colombia, Peru, and Chile."
    },
    {
        "instruction": "What is the purpose of the periodic table of elements?",
        "input": "",
        "output": "The periodic table of elements organizes all known chemical elements based on their atomic number, electron configuration, and recurring chemical properties. It serves as a useful reference for understanding element relationships and predicting their chemical behavior."
    },
    {
        "instruction": "What are the key factors that contribute to effective teamwork?",
        "input": "",
        "output": "Key factors that contribute to effective teamwork include clear communication, mutual respect, well-defined roles and responsibilities, trust among team members, and collaborative problem-solving."
    },
    {
        "instruction": "Describe the concept of ‘artificial intelligence’.",
        "input": "",
        "output": "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn like humans. AI systems can perform tasks such as recognizing speech, making decisions, and solving problems, often using algorithms and data to improve their performance over time."
    },
    {
        "instruction": "What is the significance of the Hubble Space Telescope?",
        "input": "",
        "output": "The Hubble Space Telescope has significantly advanced our understanding of the universe by providing high-resolution images and data on various celestial phenomena. It has contributed to key discoveries."
    }
]
def basic_llm(instruction): #Interact with "LLM"
    for item in instructions:
        if item["instruction"].lower() == instruction.lower():
            return item["output"]
    return "Question is beyond Instruction dataset parameters"

def main(): #Generate Instruction Sets
    instruction_set = []
    for _ in range(3000):
        instruction = random.choice(instructions)
        instruction_set.append({
            "instruction": instruction["instruction"],
            "input": instruction["input"],
            "output": basic_llm(instruction["instruction"])
        })


    with open('instruction_set.json', 'w') as f: #Create JSON file for instruction sets
        json.dump(instruction_set, f, indent=4)

def interact_llm():
    print("Use 'exit' to end the chat.")
    while True:
        question = input("Ask a me question: ")
        if question.lower() == 'end chat':
            break
        response = basic_llm(question)
        print("Response:", response)
        print()

if __name__ == "__main__": #Functions of Program
    main()
    interact_llm()

    ##def generate_data(num_data):
        ##data = []
        ##for i in range(num_data):
            ##instruct = {
                ##"instruction": f"Generated Instruction Set {i + 1}",
                ##"input": f"Generated Input {i + 1}",
                ##"output": f"Generated Output {i + 1}"
            ##}
            ##data.append(instruct)
            ##return data


    ##generate_additional_instruction = generate_data(2000)
    ##combined_instruction_sets = initial_instruction_data + generate_additional_instruction

    ##with open('instruction_set.json', 'w') as f:
        ##json.dump(combined_instruction_sets, f, indent=4)

    ##print("New Instruction Sets have been generated")