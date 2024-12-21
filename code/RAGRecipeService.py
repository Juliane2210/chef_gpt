from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from rank_bm25 import BM25Okapi  # BM25 for retrieval
import os

# Key Components of the RAG Implementation:
# Document Store (Retrieval):

# I used a simulated list of recipe texts as the "knowledge base".
# BM25 (via rank_bm25) retrieves the most relevant document(s) based on a query.
# Combining Retrieved Context:

# Retrieved documents are added as context to the GPT-2 prompt.
# Prompt Augmentation:

# GPT-2 generates text conditioned on the query and the relevant retrieved context.
# Parameters:

# top_n: Number of top documents retrieved.
# max_length: Length of GPT-2 generated text.
# temperature: Controls randomness of text generation.


# Simulated Recipe Data Store
DOCUMENTS = [
    "Recipe for chocolate chip cookies: Mix flour, sugar, chocolate chips, and bake at 350°F for 12 minutes.",
    "Peppermint chocolate cake: Combine flour, sugar, cocoa powder, and peppermint essence. Bake at 375°F for 25 minutes.",
    "Classic sugar cookies: Mix flour, sugar, butter, and vanilla extract. Bake at 350°F for 10 minutes.",
    "Mint chocolate chip brownies: Combine chocolate, peppermint, sugar, and eggs. Bake at 350°F for 20 minutes.",
    "Christmas spice gingerbread: Combine flour, molasses, ginger, cinnamon, and cloves. Bake at 350°F for 15 minutes.",
    "Eggnog sugar cookies: Mix flour, sugar, butter, and eggnog. Bake at 350°F for 12 minutes.",
    "White chocolate cranberry bars: Combine flour, sugar, dried cranberries, and white chocolate. Bake at 375°F for 18 minutes.",
    "Peppermint bark fudge: Layer melted dark chocolate, white chocolate, and crushed candy canes. Chill until set.",
    "Almond snowball cookies: Mix almond flour, butter, powdered sugar, and vanilla. Bake at 325°F for 10 minutes.",
    "Cinnamon roll cookies: Roll cinnamon sugar into cookie dough, slice, and bake at 350°F for 12 minutes.",
    "Hot cocoa brownies: Add hot cocoa mix and mini marshmallows to brownie batter. Bake at 350°F for 25 minutes.",
    "Christmas tree cupcakes: Decorate vanilla cupcakes with green frosting and sprinkles to resemble trees.",
    "Raspberry linzer cookies: Make sandwich cookies with raspberry jam and powdered sugar topping. Bake at 350°F for 10 minutes.",
    "Chocolate-dipped shortbread: Bake shortbread cookies and dip one end in melted dark chocolate.",
    "Cherry almond biscotti: Combine flour, dried cherries, and almonds. Bake at 350°F for 30 minutes.",
    "Vanilla peppermint macarons: Fill macaron shells with peppermint-flavored buttercream.",
    "Chocolate crinkle cookies: Mix cocoa powder, sugar, and flour. Roll in powdered sugar and bake at 350°F for 10 minutes.",
    "Red velvet thumbprint cookies: Make thumbprint cookies with cream cheese filling. Bake at 350°F for 12 minutes.",
    "Holiday rum balls: Mix crushed cookies, cocoa powder, and rum. Roll into balls and coat with powdered sugar.",
    "Candy cane meringues: Whisk egg whites and sugar, pipe into candy cane shapes, and bake at 200°F for 2 hours.",
    "Caramel pecan bars: Bake shortbread crust and top with caramel and pecans. Chill before serving.",
    "Chocolate orange truffles: Mix chocolate and orange zest, chill, and roll into truffle balls.",
    "Snowflake sugar cookies: Decorate sugar cookies with royal icing and silver sprinkles.",
    "Stained glass cookies: Cut shapes from cookie dough and fill with crushed candies. Bake at 350°F for 10 minutes.",
    "Gingerbread cupcakes: Bake spiced cupcakes and top with cream cheese frosting.",
    "Holiday spice muffins: Mix flour, cinnamon, nutmeg, and cloves. Bake at 375°F for 18 minutes.",
    "Maple pecan fudge: Combine maple syrup, sugar, and pecans. Chill until firm and cut into squares.",
    "Toffee chocolate bark: Layer melted chocolate and toffee pieces. Chill until set and break into pieces.",
    "Pumpkin spice cookies: Mix pumpkin puree, flour, and spices. Bake at 350°F for 12 minutes.",
    "Chocolate peppermint bark: Layer dark and white chocolate with crushed candy canes.",
    "Marshmallow snowmen: Stack marshmallows, secure with pretzel sticks, and decorate with icing.",
    "Fruitcake cookies: Mix flour, dried fruit, and nuts. Bake at 325°F for 15 minutes.",
    "Peanut butter blossoms: Bake peanut butter cookies and press a chocolate kiss on top while warm.",
    "Spiced molasses cookies: Combine molasses, flour, ginger, and cinnamon. Bake at 350°F for 12 minutes.",
    "Orange cranberry cookies: Mix orange zest, dried cranberries, and sugar. Bake at 375°F for 10 minutes.",
    "Chai latte cookies: Add chai spices to sugar cookie dough and bake at 350°F for 12 minutes.",
    "Holiday cheesecake bites: Bake mini cheesecakes and top with cranberry sauce.",
    "Brown butter toffee cookies: Add browned butter and toffee bits to cookie dough. Bake at 350°F for 10 minutes.",
    "Ginger molasses crinkles: Mix molasses, ginger, and sugar. Roll in sugar and bake at 350°F for 12 minutes.",
    "Mini fruit tarts: Fill tart shells with custard and top with sliced fruit. Chill before serving.",
    "Peppermint mocha cookies: Add espresso powder and peppermint extract to chocolate cookie dough.",
    "Yule log cake: Bake chocolate sponge cake, fill with cream, and roll into a log shape.",
    "Coconut snowball truffles: Mix coconut, condensed milk, and roll into balls. Coat with shredded coconut.",
    "Holiday chocolate pretzels: Dip pretzels in chocolate and sprinkle with festive toppings.",
    "Vanilla bean shortbread: Mix flour, sugar, butter, and vanilla bean. Bake at 325°F for 10 minutes.",
    "Spiced apple hand pies: Fill pastry dough with spiced apple filling and bake at 375°F for 15 minutes.",
    "Christmas wreath cookies: Shape cookie dough into wreaths and decorate with red and green icing.",
    "Chocolate hazelnut thumbprints: Make thumbprint cookies filled with chocolate hazelnut spread.",
    "Mint mocha brownies: Add mint extract and espresso powder to brownie batter. Bake at 350°F for 20 minutes."

]

# Initialize BM25 for retrieval
tokenized_docs = [doc.lower().split() for doc in DOCUMENTS]
bm25 = BM25Okapi(tokenized_docs)


class RAGRecipeGenerator:
    def __init__(self, model_name="gpt2-medium"):
        # Initialize GPT-2 model and tokenizer
        self.model_name = model_name
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        print("GPT-2 Model and tokenizer loaded successfully!")

    def retrieve_context(self, query, top_n=1):
        """
        Retrieve top_n relevant documents using BM25.
        """
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)),
                             key=lambda i: scores[i], reverse=True)[:top_n]
        retrieved_docs = [DOCUMENTS[i] for i in top_indices]
        return "\n".join(retrieved_docs)

    def generate_recipe(self, query, max_length=500, temperature=0.7):
        """
        Generate recipe by combining retrieved context and GPT-2 generation.
        """
        # Step 1: Retrieve relevant documents
        retrieved_context = self.retrieve_context(query, top_n=1)

        # Step 2: Combine query and retrieved context as input
        combined_prompt = f"{retrieved_context}\nUser query: {query}\nRecipe:"

        # Encode prompt
        input_ids = self.tokenizer.encode(combined_prompt, return_tensors="pt")

        # Step 3: Generate text using GPT-2
        output = self.model.generate(
            input_ids,
            max_length=input_ids.shape[1] + max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            top_k=50,
            top_p=0.9,
            temperature=temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # Decode and return generated recipe
        recipe = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return recipe


if __name__ == "__main__":
    # Initialize RAG-based generator
    generator = RAGRecipeGenerator()

    # Query with ingredients
    ingredients = ["flour", "peppermint", "sugar", "chocolate chips"]
    query = f"Recipe using {', '.join(ingredients)}"

    # Generate recipe
    print("Generating recipe...")
    recipe = generator.generate_recipe(query)
    print("\nGenerated Recipe:")
    print(recipe)
