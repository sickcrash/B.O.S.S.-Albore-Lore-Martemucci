# TASK 4 - Modello unificato indoor/outdoor con adattamento di dominio (2D→BEV “light”)
Gruppo: Bianca Di Bitetto, Francesco Salvatore Desiderato, Francesco Maria Di Cataldo

Obiettivo:  progettare un framework per addestrare un detector unico che generalizzi tra indoor e outdoor, che includa:

1. Modulo di domain alignment (normalizzazione/adaptive layers);
2. Bridge backbone 2D↔testa BEV;
3. Plugin per i metodi di adattamento di dominio;
4. Adapter per unificare loader di dataset eterogenei;
5. Valutazione cross-domain mAP e degrado fuori dominio.

I dataset utilizzati sono KITTI per dominio outdoor e SUNRGBD per dominio indoor.
