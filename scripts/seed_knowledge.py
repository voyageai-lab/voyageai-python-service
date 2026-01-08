#!/usr/bin/env python3
"""
Seed Travel Knowledge Script

This script populates both ChromaDB (vector) and Elasticsearch (BM25)
with sample travel knowledge for testing and demonstration.

Usage:
    cd voyageai-python-service
    source .venv/bin/activate
    python scripts/seed_knowledge.py
    
Prerequisites:
    - OPENAI_API_KEY in .env or environment
    - (Optional) Elasticsearch running on localhost:9200
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from voyageai.rag.chroma import chroma_client
from voyageai.rag.elasticsearch_client import es_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================================
# Sample Travel Knowledge Data
# ============================================================================

TRAVEL_KNOWLEDGE = [
    # Tokyo
    {
        "content": """The best time to visit Tokyo is during spring (late March to early May) 
for cherry blossom season, or autumn (October to November) for beautiful fall colors. 
Cherry blossom viewing (hanami) is a beloved tradition where locals and tourists gather 
in parks like Ueno and Shinjuku Gyoen to picnic under the blooming sakura trees. 
The blossoms typically peak around late March to early April, but exact timing varies yearly.""",
        "title": "Best Time to Visit Tokyo",
        "destination": "Tokyo",
        "category": "seasonal",
        "source": "tokyo_travel_guide.txt"
    },
    {
        "content": """Tokyo's public transportation is world-renowned for its efficiency. 
The JR Yamanote Line circles central Tokyo, connecting major stations like Shibuya, 
Shinjuku, Ikebukuro, and Tokyo Station. The metro system covers areas the JR doesn't reach. 
A Suica or Pasmo IC card is essential - these rechargeable cards work on all trains, 
buses, and even convenience stores. First trains run around 5:00 AM, last trains around midnight.""",
        "title": "Tokyo Transportation Guide",
        "destination": "Tokyo",
        "category": "transportation",
        "source": "tokyo_travel_guide.txt"
    },
    {
        "content": """Budget travelers can enjoy Tokyo for around ¥10,000-15,000 per day 
($70-100 USD). Stay in capsule hotels or hostels (¥3,000-5,000/night), eat at ramen shops 
and convenience stores (¥500-1,000/meal), and take advantage of free attractions like 
Senso-ji Temple, Meiji Shrine, and window shopping in Harajuku. A 7-day JR Pass costs 
around ¥50,000 but is only worth it if making day trips to other cities.""",
        "title": "Tokyo Budget Travel Tips",
        "destination": "Tokyo",
        "category": "budget",
        "source": "tokyo_travel_guide.txt"
    },
    {
        "content": """Senso-ji Temple in Asakusa is Tokyo's oldest and most famous Buddhist temple, 
founded in 628 AD. The iconic Kaminarimon (Thunder Gate) with its massive red lantern is 
a must-photograph spot. Behind it, Nakamise shopping street offers traditional snacks and 
souvenirs. The temple grounds are free to enter and especially atmospheric in the early 
morning before crowds arrive. Evening visits when the pagoda is lit up are also magical.""",
        "title": "Senso-ji Temple Visitor Guide",
        "destination": "Tokyo",
        "category": "attractions",
        "source": "tokyo_attractions.txt"
    },
    # Kyoto
    {
        "content": """Kyoto's famous Fushimi Inari Shrine features thousands of vermillion 
torii gates winding up Mount Inari. The full hike takes 2-3 hours, but most visitors 
explore just the lower sections. The shrine is dedicated to Inari, the Shinto god of rice, 
and foxes (kitsune) are considered Inari's messengers. The shrine is open 24/7 and is 
especially beautiful at dawn or dusk when the gates glow in golden light.""",
        "title": "Fushimi Inari Shrine Guide",
        "destination": "Kyoto",
        "category": "attractions",
        "source": "kyoto_guide.txt"
    },
    {
        "content": """The historic geisha district of Gion in Kyoto is the best place to 
spot geiko (Kyoto dialect for geisha) and maiko (apprentice geisha). Early evening 
around 5-6 PM offers the best chances as they head to appointments. Photography should 
be respectful - never chase or block geisha. The area's traditional machiya townhouses 
and teahouses create an atmospheric walk back in time.""",
        "title": "Gion District and Geisha Culture",
        "destination": "Kyoto",
        "category": "culture",
        "source": "kyoto_guide.txt"
    },
    {
        "content": """Autumn in Kyoto (November to early December) is arguably the most 
beautiful time to visit. The momiji (maple) leaves turn brilliant shades of red, orange, 
and gold. Top spots for fall foliage include Tofuku-ji Temple, Eikando Temple, and 
Arashiyama. Many temples offer special evening illuminations. Book accommodations 
months in advance as this is peak tourist season.""",
        "title": "Kyoto Autumn Foliage Guide",
        "destination": "Kyoto",
        "category": "seasonal",
        "source": "kyoto_guide.txt"
    },
    # Paris
    {
        "content": """The Eiffel Tower is best visited in the morning to avoid long queues. 
Book tickets online at least a week in advance for the summit access. The tower has three 
levels: first and second accessible by stairs or elevator, summit by elevator only. 
For the best photos, visit at golden hour or stay for the sparkling light show that 
happens every hour after dark. Nearby Champ de Mars offers perfect picnic spots.""",
        "title": "Eiffel Tower Visitor Tips",
        "destination": "Paris",
        "category": "attractions",
        "source": "paris_guide.txt"
    },
    {
        "content": """Paris's best croissants can be found at artisan boulangeries like 
Du Pain et des Idées, Blé Sucré, and Poilâne. Look for croissants that are golden brown, 
flaky, and slightly shiny from butter. Many bakeries bake throughout the day, but 
morning is when you'll find the freshest selection. A proper Parisian breakfast is 
simple: croissant, café, and maybe a tartine (bread with butter and jam).""",
        "title": "Paris Food: Best Croissants",
        "destination": "Paris",
        "category": "food",
        "source": "paris_food_guide.txt"
    },
    {
        "content": """The Louvre requires at least 3-4 hours to see the highlights: 
Mona Lisa, Venus de Milo, and Winged Victory of Samothrace. Wednesday and Friday 
evenings (until 9:45 PM) are less crowded. Enter through the Porte des Lions entrance 
instead of the main pyramid to skip lines. Download the free Louvre app for navigation. 
The museum is closed on Tuesdays.""",
        "title": "Louvre Museum Visitor Guide",
        "destination": "Paris",
        "category": "attractions",
        "source": "paris_guide.txt"
    },
    # New York
    {
        "content": """Central Park is best explored on foot or by renting a bike from 
Citi Bike. Don't miss Bethesda Terrace, Bow Bridge, and the Conservatory Garden. 
In winter, ice skating at Wollman Rink is iconic. The park is safest during daylight 
hours; stick to busy paths after dark. Free events include Shakespeare in the Park 
(summer) and concerts at the Great Lawn.""",
        "title": "Central Park Complete Guide",
        "destination": "New York",
        "category": "attractions",
        "source": "nyc_guide.txt"
    },
    {
        "content": """New York pizza is a must-try. Iconic spots include Joe's Pizza 
in Greenwich Village, Di Fara in Brooklyn, and Lucali for special occasions. 
New York-style pizza is characterized by its thin, wide slices that you fold in half 
to eat. A $1 slice is part of the NYC experience, but quality can vary. For late-night 
cravings, many pizzerias stay open until 2-4 AM.""",
        "title": "NYC Pizza Guide",
        "destination": "New York",
        "category": "food",
        "source": "nyc_food_guide.txt"
    },
    # Bangkok
    {
        "content": """Bangkok's Grand Palace is a must-see, but dress code is strictly 
enforced: long pants (no shorts), covered shoulders, closed shoes. The complex 
includes Wat Phra Kaew (Temple of the Emerald Buddha), Thailand's most sacred temple. 
Visit early (opens 8:30 AM) to beat the heat and crowds. Allow 2-3 hours. Beware of 
scammers outside who claim the palace is closed - it's never closed unannounced.""",
        "title": "Grand Palace Visitor Guide",
        "destination": "Bangkok",
        "category": "attractions",
        "source": "bangkok_guide.txt"
    },
    {
        "content": """Street food in Bangkok is legendary. Yaowarat (Chinatown) comes 
alive after 6 PM with dozens of food stalls. Must-try dishes include pad thai, som tam 
(papaya salad), mango sticky rice, and boat noodles. Look for busy stalls with high 
turnover - that's where the freshest food is. Budget about ฿50-150 per dish. Bring cash 
as most vendors don't accept cards.""",
        "title": "Bangkok Street Food Guide",
        "destination": "Bangkok",
        "category": "food",
        "source": "bangkok_food_guide.txt"
    },
    # General Travel Tips
    {
        "content": """Jet lag can be minimized by adjusting your sleep schedule a few days 
before departure. Stay hydrated during flights, avoid alcohol, and try to sleep on the 
plane if arriving in the morning. Upon arrival, get sunlight exposure to reset your 
circadian rhythm. Melatonin supplements can help, especially for eastward travel. 
The general rule: it takes about one day per time zone crossed to fully adjust.""",
        "title": "Managing Jet Lag",
        "destination": "General",
        "category": "tips",
        "source": "travel_tips.txt"
    },
    {
        "content": """Travel insurance is essential for international trips. Look for 
policies covering medical emergencies (at least $100,000), trip cancellation, lost 
luggage, and emergency evacuation. Read the fine print for pre-existing condition 
exclusions. Keep copies of your policy and emergency numbers both digitally and printed. 
Popular providers include World Nomads, Allianz, and travel credit card benefits.""",
        "title": "Travel Insurance Guide",
        "destination": "General",
        "category": "tips",
        "source": "travel_tips.txt"
    },
]


async def seed_database():
    """Seed both vector and BM25 databases with travel knowledge."""
    
    logger.info("Starting database seeding...")
    
    # Initialize clients
    await chroma_client.initialize()
    es_available = await es_client.initialize()
    
    # Extract content and metadata
    contents = [doc["content"] for doc in TRAVEL_KNOWLEDGE]
    metadatas = [
        {
            "title": doc["title"],
            "destination": doc["destination"],
            "category": doc["category"],
            "source": doc["source"],
        }
        for doc in TRAVEL_KNOWLEDGE
    ]
    
    # Seed ChromaDB
    logger.info(f"Indexing {len(contents)} documents to ChromaDB...")
    vector_count = await chroma_client.add_documents(
        documents=contents,
        metadatas=metadatas,
    )
    logger.info(f"Indexed {vector_count} documents to ChromaDB")
    
    # Seed Elasticsearch if available
    if es_available:
        logger.info(f"Indexing {len(TRAVEL_KNOWLEDGE)} documents to Elasticsearch...")
        bm25_count = await es_client.add_documents(TRAVEL_KNOWLEDGE)
        logger.info(f"Indexed {bm25_count} documents to Elasticsearch")
    else:
        logger.warning("Elasticsearch not available, skipping BM25 indexing")
    
    # Print summary
    logger.info("=" * 50)
    logger.info("Seeding complete!")
    logger.info(f"  ChromaDB: {chroma_client.count()} documents")
    if es_available:
        logger.info(f"  Elasticsearch: available")
    else:
        logger.info(f"  Elasticsearch: not available (run docker-compose up)")
    logger.info("=" * 50)


async def test_search():
    """Test search after seeding."""
    
    logger.info("\nTesting search...")
    
    from voyageai.rag.hybrid_search import hybrid_search
    
    test_queries = [
        "best time to visit Tokyo",
        "where to find good pizza in New York",
        "Kyoto geisha district",
    ]
    
    for query in test_queries:
        logger.info(f"\nQuery: '{query}'")
        result = await hybrid_search(
            query=query,
            top_k=2,
            rerank=False,  # Skip reranking for quick test
        )
        
        for i, r in enumerate(result.results):
            title = r.get("metadata", {}).get("title", "N/A")
            logger.info(f"  {i+1}. {title} (score: {r.get('rrf_score', 'N/A')})")


if __name__ == "__main__":
    async def main():
        await seed_database()
        await test_search()
    
    asyncio.run(main())

