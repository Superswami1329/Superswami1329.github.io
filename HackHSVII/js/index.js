var text = ["C", "Co", "Cod", "Code", "Coder", "Coders", "Coders.", "Coders.","Coders.","Coders.","Coders.","Coders.","Coders.","Coders.","Coders.", "Coders", "Coder", "Code", "Cod", "Co", "C",

      "C", "Cr", "Cre", "Crea", "Creat", "Creato", "Creator", "Creators", "Creators.", "Creators.", "Creators.", "Creators.", "Creators.", "Creators.", "Creators.", "Creators.", "Creators.", "Creators", "Creator", "Creato", "Creat", "Crea", "Cre", "Cr", "C",

      "I", "In", "Inn", "Inno", "Innov", "Innova", "Innovat", "Innovato", "Innovator", "Innovators", "Innovators.", "Innovators.", "Innovators.", "Innovators.", "Innovators.", "Innovators.", "Innovators.", "Innovators.", "Innovators.", "Innovators", "Innovator", "Innovato", "Innovat", "Innova", "Innov", "Inno", "Inn", "In", "I",

      "D", "Dr", "Dre", "Drea", "Dream", "Dreame", "Dreamer", "Dreamers", "Dreamers.", "Dreamers.", "Dreamers.", "Dreamers.", "Dreamers.", "Dreamers.", "Dreamers.", "Dreamers.", "Dreamers.", "Dreamers", "Dreamer", "Dreame", "Dream", "Drea", "Dre", "Dr", "D",

      "Y", "YO", "YOU", "YOU.", "YOU.", "YOU.", "YOU.", "YOU.", "YOU.", "YOU.", "YOU.", "YOU.", "YOU.",  "YOU.", "YOU.", "YOU.",  "YOU", "YO", "Y"];
    var counter = 0;
    setInterval(change, 100);
    function change() {
     elem.innerHTML = text[counter];
        counter++;
        if(counter >= text.length) { counter = 0; }
    }