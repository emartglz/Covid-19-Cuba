from corona import read_days, read_json

def scan_municipies(days):
    municpies = {}

    for i in range(len(days)):
        try:
            l = len(days[str(i + 1)]['diagnosticados'])

            for j in range(l):
                try:
                    if days[str(i + 1)]['diagnosticados'][j]['provincia_detección'] == 'La Habana':
                        municpies[days[str(i + 1)]['diagnosticados'][j]['municipio_detección']] = 1
                except:
                    pass
        except:
            pass

    print (list(municpies))

def main():
    data = read_json('data.json')

    assert data != None, 'ERROR READING FILE'

    days = read_days(data)
    scan_municipies(days)



if __name__ == '__main__':
    main()
    