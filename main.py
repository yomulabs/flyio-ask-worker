from dotenv import load_dotenv
import pinecone
import os
from sentence_transformers import SentenceTransformer
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from timeit import default_timer as timer
import requests
from bs4 import BeautifulSoup


load_dotenv()

MODEL_CHUNK_SIZE = 512

pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))

index = pinecone.Index("fly-io-docs")

# https://huggingface.co/spaces/mteb/leaderboard
# https://huggingface.co/embaas/sentence-transformers-e5-large-v2
model = SentenceTransformer('embaas/sentence-transformers-e5-large-v2')

def add_html_to_vectordb(content, path):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = MODEL_CHUNK_SIZE,
        chunk_overlap  = 20
    )

    docs = text_splitter.create_documents([content])


    for doc in docs:
        insert_vector(doc.page_content, path)

def rebuild_vector_for_path(path):
    print("eh")

def insert_vector(text, path):
    start = timer()
    embedding = model.encode(text)
    end = timer()
    print(f'encode took {end - start} seconds')

    last_id  = index.describe_index_stats()['total_vector_count']

    upserted_data = []
    vector = (str(last_id + 1), embedding.tolist(), { 'content': text, 'path': path })
    upserted_data.append(vector)

    index.upsert(vectors=upserted_data)
    end = timer()

    print(f'insert_vector {len(text)} took {end - start} seconds')

def list_vectors():
    results = index.fetch(ids=[])
    print(results)

def query_vector(user_query):
    query_em = model.encode(user_query).tolist()
    result = index.query(query_em, top_k=3, includeMetadata=True)
    return result

def ask_chatgpt(knowledge_base, user_query):
    system_content = """You are an AI coding assistant designed to help users with their programming needs based on the Knowledge Base provided.
    If you dont know the answer, say that you dont know the answer. You will only answer questions related to fly.io, any other questions, you should say that its out of your responsibilities.
    """

    user_content = f"""
        Knowledge Base:
        ---
        {knowledge_base}
        ---
        User Query: {user_query}
        Answer:
    """
    system_message = {"role": "system", "content": system_content}
    user_message = {"role": "user", "content": user_content}
    print(user_content)
    chatgpt_response = openai.ChatCompletion.create(model="gpt-3.5-turbo-0613", messages=[system_message, user_message])
    print(chatgpt_response)

def run():
    #user_query = "how do i scale cpu?"
    user_query = "automatically stop machines? is it possible?"
    result = query_vector(user_query)
    list_of_knowledge_base = map(lambda match: match.metadata['content'], result.matches)
    list_of_sources = map(lambda match: match.metadata['path'], result.matches)

    print("sources")
    for source in list_of_sources:
        print(source)

    knowledge_base = "\n".join(list_of_knowledge_base)
    knowledge_base = """
Search\n⌘K\nGetting Started\nSpeedrun: Deploying an App\nHands-on with Fly.io\nWorking with Fly Apps\nTroubleshooting Deployments\nConnect to an App Service\nWeb Launchers\nLanguage & Framework Guides\nRun an Elixir App\nRun a Rails App\nRun a Laravel App\nRun a Django App\nMore...\nFly Apps\nLaunch a New App\nDeploy a Fly App\nGet Information about an App\nAdd Volume Storage\nScale Machine CPU and RAM\nScale the Number of Machines\nAuto Stop and Start Machines\nRestart an App\nRun Multiple Processes in an App\nDelete an App\nScale V1 (Nomad) Apps\nMigrate an Existing Fly App to Apps V2\nApp Configuration Reference\nDatabases & Storage\nVolumes\nFly Postgres\nSQLite & LiteFS\nRedis by Upstash\nMore...\nFly Machines\nWorking with the Machines API\nRun Machines with flyctl\nMachine Sizing\nRun User Code on Fly Machines\nUse Terraform with Fly Machines\nOther Guides\nGoing to Production\nCustom Domains for SaaS\nDeploy with GitHub Actions\nRun Multiple Processes\nRun UDP Services\nCrontab with Supercronic\nFly.io Reference\nflyctl\nApp Configuration (fly.toml)\nApps\nArchitecture\nAvailability and Resiliency\nBuilders\nDeploy Tokens\nDynamic Request Routing\nFly Launch\nLoad Balancing\nMachines\nMetrics\nMonorepo Apps\nPrivate Networking\nPublic Networking\nRegions\nRuntime Environment\nBuild Secrets\nRuntime Secrets\nTLS Support\nVolumes\nAbout\nPricing\nHow We Use Credit Cards\nEngineering Jobs\nHealthcare on Fly.io\nSupport\nSecurity\nExtensions Program\nOpen Source\nUsing Our Brand\nPrivacy Policy\nTerms of Service\nAutomatically Stop and Start Machines\nThis feature only works for V2 apps running on Fly Machines. You might also be interested in learning about scaling the number of machines for the V2 Apps Platform. For information about scaling V1 apps, refer to Scale V1 Nomad Apps.\n\nFly Machines are fast to start and stop, and you don't pay for their CPU and RAM when they're in the stopped state. For Fly Apps with a service configured, Fly Proxy can automatically start and stop existing Machines based on incoming requests, so that your app can meet demand without keeping extra Machines running. And if your app needs to have one or more Machines always running in your primary region, then you can set a minimum number of machines to keep running.\n\nThis Fly Proxy feature also plays well with apps whose Machines exit from within when idle. If your app already shuts down when idle, then the proxy can restart it when there's traffic.\n\nConfigure Automatic Start and Stop\n\nThe autostart and stop settings apply per service, so you set them within the [[services]] or [http_service] sections of fly.toml. You can configure automatic starts and stops separately with the auto_start_machines and auto_stop_machines settings, and set the minimum number of machines to keep running with the min_machines_running setting.\n\nConcurrency limits for services affect how automatic starts and stops work.\n\nDefault and Recommended Values\n\nDefault settings in fly.toml for V2 apps created using the fly launch command:\n\n...\n[[services]]\n  internal_port = 8080\n  protocol = "tcp"\n  auto_stop_machines = true\n  auto_start_machines = true\n  min_machines_running = 0\n...\n\n...\n[http_service]\n  internal_port = 8080\n  force_https = true\n  auto_stop_machines = true\n  auto_start_machines = true\n  min_machines_running = 0\n...\n\nExisting V2 apps—or any V2 apps that don't have these settings in fly.toml—have the default values auto_start_machines = true and auto_stop_machines = false.\n\nIn general, we recommend setting auto_stop_machines and auto_start_machines to the same value to avoid having Machines that either never start or never stop.\n\nIf auto_start_machines = true and auto_stop_machines = false, then Fly Proxy will automatically start your Machines but will never stop them. This means you'll have to stop Machines manually, or have your app exit when idle.\n\nIf auto_start_machines = false and auto_stop_machines = true, then Fly Proxy will automatically stop your Machines when there's low traffic, but won't be able to start them again. If all or most of your Machines are stopped, then requests to your app will start failing.\n\nYou can set min_machines_running to 1 or higher if you need at least one instance of your app running all the time. This setting is for the total number of Machines, not Machines per region. For example, if min_machines_running = 1, then your app will scale down until there is only one Machine running in your primary region.\n\nThere's no "maximum machines running" setting, because the maximum number of Machines is just the total number of Machines you've created for your app. Learn more in the How It Works section.\n\nHow It Works\n\nThe Fly Proxy runs a process to automatically stop and start existing Fly Machines every few minutes.\n\nThe automatic start and stop feature only works on existing Machines and never creates or destroys Machines for you. The maximum number of running Machines is the number of Machines you've created for your app using fly scale count or fly machine clone. Learn more about scaling the number of Machines.\nFly Proxy Stops Machines\n\nWhen auto_stop_machines = true in your fly.toml, the proxy looks at Machines running in a single region and uses the concurrency soft_limit setting for each Machine to determine if there's excess capacity. If the proxy decides there's excess capacity, it stops exactly one machine. The proxy repeats this process every few minutes, stopping only one machine per region, if needed, each time.\n\nIf you have the kill_signal and kill_timeout options configured in your fly.toml file, then Fly Proxy uses those settings when it stops a Machine.\n\nFly Proxy determines excess capacity per region as follows:\n\nIf there's more than one Machine in the region:\nthe proxy determines how many running Machines are over their soft_limit setting and then calculates excess capacity: excess capacity = num of machines - (num machines over soft limit + 1)\nif excess capacity is 1 or greater, then the proxy stops one Machine\nIf there's only one Machine in the region:\nthe proxy checks if the Machine has any traffic\nif the Machine has no traffic (a load of 0), then the proxy stops the Machine\nFly Proxy Starts Machines\n\nWhen auto_start_machines = true in your fly.toml, the Fly Proxy restarts a Machine in the nearest region when required.\n\nFly Proxy determines when to start a Machine as follows:\n\nThe proxy waits for a request to your app.\nIf all the running Machines are above their soft_limit setting, then the proxy starts a stopped Machine in the nearest region (if there are any stopped Machines).\nThe proxy routes the request to the newly started Machine.\nWhen to Stop and Start Fly Machines Automatically, or Not\n\nIf your app has highly variable request workloads, then you can set auto_stop_machines and auto_start_machines to true to manage your Fly Machines as demand decreases and increases. This could reduce costs, because you'll never have to run excess Machines to handle peak load; you'll only run, and be charged for, the number of Machines that are needed at any given time.\n\nThe difference between this feature and what is typical in autoscaling, is that it doesn't create new Machines up to a specified maximum. It automatically starts only existing Machines. For example, if you want to have a maximum of 10 Machines available to service requests, then you need to create 10 Machines for your app.\n\nIf you need all of your app's Machines to be “always on”, then you can set auto_stop_machines and auto_start_machines to false. If auto_stop_machines = true, min_machines_running = 0, and there’s no traffic to your app, eventually all of your app's Machines could be stopped.\n\nIf you only need one or a few instances of your app to keep running in your primary region all the time, then you can set min_machines_running to 1 or higher.\n\nStop a Machine by Terminating Its Main Process\n\nSetting your app to automatically stop when there's excess capacity using auto_stop = true is a substitute for when your app doesn't implement automatic shut down after a period of inactivity. If you want a custom shut-down process for your app, then you can code your app to exit from within when idle.\n\nHere are some examples:\n\nShutting Down a Phoenix App When Idle: a post by Chris McCord on adding a task to an Elixir app's supervision tree that shuts down the Erlang runtime when there are no active connections.\nFor Rails apps, dockerfile-rails provides a --max-idle option that will exit after n seconds of inactivity.\nA Tired Proxy in Go used in Building an In-Browser IDE the Hard Way. There's a community fork with more recent updates.\nA minimal demo app in Typescript/Remix: code & demo.\n\nAs of flyctl v0.0.520, Fly Postgres supports this too!\n\nEdit this page on GitHub
    """
    ask_chatgpt(knowledge_base, user_query)


def get_html_body_content(url):
    response = requests.get(url)

    # Parse the HTML content
    soup = BeautifulSoup(response.content, "html.parser")

    # Find the body element and extract its inner text
    body = soup.body
    inner_text = body.get_text()
    return inner_text

def get_html_sitemap(url):
    response = requests.get(url)

    # Parse the HTML content
    soup = BeautifulSoup(response.content, "lxml")

    # Find the body element and extract its inner text
    links = []

    locations = soup.find_all("loc")
    for location in locations:
        url = location.get_text()
        if "fly.io/docs" in url:
            links.append(url)

    return links


def index_website():
    links = get_html_sitemap("https://fly.io/sitemap.xml")
    for link in links:
        content = get_html_body_content(link)
        add_html_to_vectordb(content, link)

run()
