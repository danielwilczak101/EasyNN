from EasyNN.scrape.scrape import google_images

google_images(
    search_term= "ship",
    driver_path='./EasyNN/scrape/chromedriver',
    target_path='./downloaded',
    number_images=5,
) 