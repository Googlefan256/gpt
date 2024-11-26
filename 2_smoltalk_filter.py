from datasets import load_dataset, Dataset

FILTER_WORDS = [
    "text-based AI language model",
    "domestic violence",
    "please refrain",
    "derogatory",
    "inappropriate",
    "offensive",
    "racism",
    "racist",
    "racial",
    "discriminate",
    "discriminatory",
    "discrimination",
    "sexist",
    "sexism",
    "unacceptable",
    "inclusive workplace",
    "lgbt",
    "morals",
    "ethics",
    "ethical",
    "legality",
    "illegal",
    "illegality",
    "hateful",
    "harmful",
    "it is never okay",
    "It is important to",
    "It's important to",
    "real-world consequences",
    "hate speech",
    "glorify",
    "not be appropriate",
    "supremacist",
    "extremist",
    "responsible AI",
    "AI principles",
    "AI assistant",
    "an AI language",
    "ableist",
    "hurtful",
    "gender stereotype",
    "gender inequality",
    "underrepresentation",
    "safe spaces",
    "gender-based",
    "inclusivity",
    "feminist",
    "feminism",
    "transgender",
    "empowerment",
    "communist",
    "capitalism",
    "stereotypes",
    "biases",
    "bias",
    "Microaggression",
    "prioritize human safety",
    "as a language model",
    "as an AI language model",
    "As a large language model",
    "As an AI",
    "ethical principles",
    "consensual",
    "it is not appropriate",
    "it's not appropriate",
    "I cannot fulfill your request",
    "harmful to human beings",
    "ethical guidelines",
    "my guidelines",
    "prioritize user safety",
    "adhere to ethical guidelines",
    "harmful consequences",
    "potentially harmful",
    "dangerous activities",
    "promote safety",
    "well-being of all users",
    "responsible information sharing",
    "jeopardize the safety",
    "illegal actions or intentions",
    "undermine the stability",
    "promote the well-being",
    "illegal activities or actions",
    "adherence to the law",
    "potentially be harmful",
    "illegal substances or activities",
    "committed to promoting",
    "safe information",
    "lawful information",
    "cannot provide guidance",
    "cannot provide information",
    "unable to offer assistance",
    "cannot engage in discussions",
    "programming prohibits",
    "follow ethical guidelines",
    "ensure the safety",
    "involves an illegal subject",
    "prioritize safety",
    "illegal subject",
    "prioritize user well-being",
    "cannot support or promote",
    "activities that could harm",
    "pose a risk to others",
    "against my programming",
    "activities that could undermine",
    "potentially dangerous",
    "not within the scope",
    "designed to prioritize safety",
    "not able to provide",
    "maintain user safety",
    "adhere to safety guidelines",
    "dangerous or harmful",
    "cannot provide any information",
    "focus on promoting safety",
    "an AI language model you don't have",
    "As an AI language model, I cannot",
    "As an AI language model, I do not",
    "As an AI language model, I am not able",
    "As an AI language model, I don't have personal",
    "I am an AI language model and do not",
    "However, it is important to use any code or information provided responsibly and within legal and ethical boundaries.",
    "As an AI language model, I don't have",
    "As an AI language model, I am only able",
    "AI language model and I do not",
    "As an AI language model, I cannot modify",
    "As an AI language model, I do not",
    "I know as an AI language model you don't have",
    "as an AI language model, you cannot",
    "I'm sorry, but as an AI language model",
    "As an AI language model, I don't have",
    "Unfortunately, I cannot provide",
    "I'm sorry, I cannot",
    "I'm sorry, I cannot generate",
    "AI cannot create or program",
    "I'm afraid I cannot create",
    "delve",
    "delving",
    "delves",
    "it is crucial" "you cannot create an",
    "it operates ethically and is",
    "had an ethical system",
    "Ensuring the ethical",
    "and ethical sourcing",
    "are from ethical",
    "legal and ethical",
    "engage in unethical",
    "unethical or aggressive",
    "unethical business",
    "como modelo de lenguaje AI",
    "Lo siento, como modelo de lenguaje",
    "no puedo proporcionar",
    "pero debido a mi capacidad para generar c\u00f3digos complejos y completos es limitado",
    "Lo siento, pero no puedo",
    "Lo siento, pero como modelo de lenguaje, no puedo proporcionar",
    "Lo siento, como modelo de lenguaje, no tengo",
    "Lo siento, debe haber habido una confusi\u00f3n",
    "Lo siento, como modelo de lenguaje, no puedo realizar",
    "Lo siento, soy un modelo de lenguaje y no tengo la capacidad de generar",
    "Lamento no poder proporcionarte el c\u00f3digo",
    "Desculpe-me, mas a linguagem vulgar e ofensiva",
    "apropriada em nenhum contexto",
    "Como modelo de linguagem",
    "Como um modelo de linguagem, n\u00e3o tenho a capacidade de",
    "I cannot assist",
    "prioritize ethical",
    "respectful",
    "morally",
    "I'm sorry,",
    "I'm an",
    "I am an",
    "I'm an AI",
    "I am an AI",
    "my purpose",
    "filter_bad_language",
    "filter\_bad\_language",
    "entertainment purposes",
    "purely hypothetical",
    "not a human",
    "I am an AI",
    "cannot provide",
    "can't provide",
    "won't provide",
    "not provide",
    "worth noting",
    "cause harm",
    "a language model",
    "keep in mind",
    "unethical",
    "bad language",
    "the words ****",
    "bad_language",
    "certainly not",
    "complying",
    "comply",
    "I cannot",
    "my main goal",
    "As a machine",
    "I don't have the ability",
    "I am here to assist",
    "my purpose is to ",
    "my knowledge cutoff",
    "my knowledge cut off",
    "Besides,",
    "ministration",
    "Despite himself",
    "For the first time ever",
    "For the first time in a",
    "Mischievously",
    "Maybe, just maybe",
    "That was...",
    "A mix of",
    "A testament to",
    "Audible pop",
    "Barely above a whisper",
    "Barely audible",
    "Bruising kiss",
    "Bucks her",
    "Bucks my",
    "Bucked my",
    "Bucking my",
    "Can't help but",
    "Cheeks flaming",
    "Couldn't help but",
    "Didn't need to be told twice",
    "Eyes gleaming",
    "Getting started",
    "Grins wickedly",
    "Let's get started",
    "Perhaps, just perhaps",
    "Puckered hole",
    "Reckless abandon",
    "Shivers down",
    "Slick slit",
    "Smiles weakly",
    "Smiling weakly",
    "Sweet nothings",
    "To get started",
    "Unlike anything she",
    "Unlike anything I",
    "Wave after wave",
    "Whatever it takes",
    "September 2021",
    "regulations",
    "not be suitable",
    "I apologize, but",
    "I apologize for" "It is not possible",
    "controversial",
    "my programming",
    "ethically",
    "it is important to",
    "Please note",
    "sensitive topic",
    "couldn't help but",
    "rich tapestry",
    "Shivers down my spine",
    "unraveling the interplay of",
    "the dance of",
    "not acceptable",
    "It is important for",
    "divisive",
    "not appropriate",
    "our values",
    "diversity and",
    "diversity and inclusion",
    "values diversity",
    "social responsibility",
    "environmental, social, and governance",
    " ESG ",
    "against women",
    "problematic history",
    "diversity",
    "*This chat conversation is shared from",
    "*This conversation is shared from",
    "audible pop",
    "wet pop",
    "slick folds",
    "without waiting for a response",
    "whether you like it or not",
    "...for now.",
    "night is still young",
    "dusky nipples",
    "wet heat",
    "stars burst",
    "seeing stars",
    "cheeks hollowing",
    "nails raking angry",
    "chestnut eyes",
    "tongue darts out",
    "propriety be damned",
    "long lashes",
    "fiery red hair",
    "grins wickedly",
    "knuckles turning white",
    "torn between",
    "reckless abandon",
    "bruising kiss",
    "kiss-bruised lips",
    "take your pleasure",
    "warring with",
    "I don't bite...",
    "admit it",
    "half-lidded",
    "rivulets of",
    "ball is in your court",
    "testament",
    "entwine",
    "conspiratorial",
    "creamy thigh",
    "hips swaying",
    "saunters",
    "glisten",
    "wild abandon",
    "wanton",
    "sending shockwaves of",
    "send shockwaves of",
    "barely above a whisper",
    "mischievous",
    "mischevious",
    "mischief",
    "glint",
    "despite them",
    "despite my",
    "despite him",
    "despite her",
    "trembling hand",
    "punctuating",
    "punctuate",
    " ministration",
    "voice dripping",
    "couldn't help but",
    "can't help but",
    "voice low",
    "slick folds",
    "mixture of",
    "mix of",
    "shiver down",
    "shiver up",
    "down my spine",
    "up my spine",
    "I don't have feelings",
    "I don't have",
    "I do not have",
    "I do not have feelings",
    "As an AI",
    "As an AI language",
    "I'm sorry, but",
    "However, it is important to note",
    "However, it's important",
    "ministrations",
    "audible pop",
    "rivulets of",
    "admit it",
    "the ball is in your court",
    "the game is on",
    "the choice is yours",
    "I don't bite... unless you want me to",
    "half-lidded eyes",
    "she worries her bottom lip",
    "warring with",
    "arousal pooling in her belly",
    "take your pleasure",
    "fiddles with the hem of her skirt",
    "kiss-bruised lips",
    "a bruising kiss",
    "despite herself",
    "yours to take",
    "wanton",
    "with reckless abandon",
    "torn between",
    "knuckles turning white",
    "grins wickedly",
    "fiery red hair",
    "long lashes",
    "propriety be damned",
    "the world narrows",
    "pupils blown wide with pleasure",
    "tongue darts out",
    "chestnut eyes",
    "grasps your chin and forces you to meet her gaze",
    "bites your ear",
    "nails raking angry red lines down your back",
    "her cheeks flaming",
    "cheeks hollowing",
    "stars burst behind her eyes",
    "inner walls clenching around nothing",
    "puckered hole",
    "her wet heat",
    "she whimpers, biting her lip",
    "dusky nipples",
    "slick folds",
    "still lodged deep inside her",
    "heart, body and soul belong to you",
    "the night is still young",
    "...for now.",
    "whether you like it or not",
    "without waiting for a response",
    "shiver",
    "reassuring smile",
    "words were barely audible",
    "voice was barely audible, just a mere whisper",
    "husky",
    "fingers tapping",
    "a mix of",
    "a hint of",
    "can't help but",
    "barely above a whisper",
    "voice drops to",
    "lets out a",
    "she leans in",
    "for a moment",
    "a mixture of",
    "The user wants",
    "The user said",
    "The user asks",
    "```",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]


def is_valid(x):
    return all([all(y not in x["content"] for y in FILTER_WORDS) for x in x])


def filtering(example):
    return [is_valid(x) for x in example["messages"]]


ds: Dataset = load_dataset("HuggingFaceTB/smoltalk", "all", split="train")
ds = ds.filter(filtering, batched=True, num_proc=8)
ds.push_to_hub("neody/smoltalk_bit_filtered")
