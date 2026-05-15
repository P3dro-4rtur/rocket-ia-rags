1 - Criar conta na AWS (usar o free tier)

2 - Baixar o Docker

3 - Fazer o Builder Docker:

Para buildar deve rodar o comando

docker build --platform linux/x86_64 -t simplerag .

4 - Baixar o AWS CLI para executar linhas de comando

5 - Usar o comando:

aws configure

e passar os paramentros necessarios

Gerar access key no IAM dentro da AWS

parametros aws configure:

AWS ACCESS KEY ID: Access key id dentro do IAM
AWS SECRET ACCESS KEY: no mesmo lugar
Default Region name: Ohio ou SP
Default output format: Pode deixar None

6 - ECR na AWS: criar repositório e ir em push container, que terá um passo a passo
com linhas de comando de acordo com o sistema operacional

7 - Lambda na AWS: selecionar container image na tela de criação da function
e seguir preenchendo os campos (não precisa mexer nas configurações avançadas).

- Mudar o timeout nas configurações da lambda para garantir que o tempo de respota vai estar dentro do possível para a llm.

- Criar a variavel de ambiente (API KEY OPENAI) na plataforma da aws na lambda function.

- Criar um test event para simular a requisição e poder observar o log dentro da plataforma da aws

8 - Disponibilizar como API criando um Load Balancers | EC2 dentro da plataforma da aws

- create load balancer
- selecionar o tipo "application load balancer"
  config:
  scheme: internet-facing
  load balancer ip address type: ipv4
  availability zones: selecionar todas as check box
  security group: default
  target group: selecionar lambda
